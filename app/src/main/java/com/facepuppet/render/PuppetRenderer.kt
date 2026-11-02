package com.facepuppet.render

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix
import android.os.SystemClock
import android.util.Log
import com.facepuppet.face.FaceFrame
import com.facepuppet.io.Recorder
import com.facepuppet.rig.AlignmentSolver
import com.facepuppet.rig.CanonicalMesh
import com.facepuppet.rig.DeformRig
import com.facepuppet.rig.FaceRig
import com.facepuppet.rig.MeshBuilder
import com.facepuppet.rig.MeshData
import com.facepuppet.rig.PhotoRig
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Thread-safe hand-off between the camera/inference threads and the GL render thread.
 *
 * * [latestFrame] is latest-wins: an `AtomicReference` that the analyzer overwrites. Stale frames
 *   are simply never read, so there is no queue and no latency growth.
 * * [rig] carries a freshly ingested photo. The renderer notices identity changes and uploads the
 *   textures/VBOs on its own thread.
 */
class PuppetState {
    val latestFrame = AtomicReference<FaceFrame>(FaceFrame.empty())
    val rig = AtomicReference<PhotoRig?>(null)
    val recenterRequested = AtomicBoolean(false)
    val snapshotRequested = AtomicBoolean(false)

    /** Written by [PuppetRenderer] every 30 frames; shown in the debug badge. */
    @Volatile
    var fps: Float = 0f

    /** Delivered on the GL thread; the UI implementation saves the PNG off-thread. */
    @Volatile
    var snapshotDelivery: ((IntArray, Int, Int) -> Unit)? = null

    /** Notified on the GL thread when recording starts/stops; the label is the saved location. */
    @Volatile
    var recordingStateListener: ((Boolean, String?) -> Unit)? = null

    /** Supplies the EGLConfig chosen by the GLSurfaceView, needed to create the encoder surface. */
    @Volatile
    var eglConfigProvider: (() -> android.opengl.EGLConfig?)? = null
}

/** Uniform locations, resolved once per program link. */
class LayerPrograms(
    val solid: Int,
    val gradient: Int,
    val overlay: Int,
    val uSolidZ: Int,
    val uSolidColor: Int,
    val uGradZ: Int,
    val uGradTop: Int,
    val uGradBottom: Int,
    val uOverlayZ: Int,
    val uOverlayTex: Int,
    val uOverlayTint: Int,
)

/** GL texture names for the layer artwork. A name of 0 means "unavailable". */
class LayerTextures(
    val teethUpper: Int,
    val teethLower: Int,
    val tongue: Int,
    val disc: Int,
    val irisLeft: Int,
    val irisRight: Int,
)

/**
 * Explicit NDC depths for the overlay passes. `-1` is the near plane, so a *smaller* number is
 * nearer to the camera: the negative Z bias pushes each layer in front of the mesh it decorates.
 */
object LayerDepth {
    const val BACKGROUND = 0.999f
    const val MASK = -0.99f
    const val INTERIOR = -0.90f
    const val TONGUE = -0.93f
    const val TEETH = -0.96f
    const val EYE = -0.97f
}

/** Dynamic VBO of NDC triangles (2 floats per vertex), no texture coordinates. */
class TriBatch(private val capacityVertices: Int) {
    private val scratch = FloatArray(capacityVertices * 2)
    private val direct: FloatBuffer =
        ByteBuffer.allocateDirect(capacityVertices * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private var vbo = 0
    private var count = 0

    fun create() {
        val names = IntArray(1)
        GLES30.glGenBuffers(1, names, 0)
        vbo = names[0]
    }

    fun destroy() {
        if (vbo != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
            vbo = 0
        }
    }

    fun clear() {
        count = 0
    }

    fun add(x: Float, y: Float) {
        if (count + 2 > scratch.size) return
        scratch[count++] = x
        scratch[count++] = y
    }

    fun draw() {
        if (vbo == 0 || count == 0) return
        direct.clear()
        direct.put(scratch, 0, count)
        direct.flip()
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, count * 4, direct, GLES30.GL_DYNAMIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 8, 0)
        GLES30.glDisableVertexAttribArray(1)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, count / 2)
    }
}

/** Dynamic VBO of NDC triangles with 2 floats of position + 2 floats of texture coordinate. */
class QuadBatch(private val capacityVertices: Int) {
    private val scratch = FloatArray(capacityVertices * 4)
    private val direct: FloatBuffer =
        ByteBuffer.allocateDirect(capacityVertices * 4 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private var vbo = 0
    private var count = 0

    fun create() {
        val names = IntArray(1)
        GLES30.glGenBuffers(1, names, 0)
        vbo = names[0]
    }

    fun destroy() {
        if (vbo != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
            vbo = 0
        }
    }

    fun clear() {
        count = 0
    }

    fun add(x: Float, y: Float, u: Float, v: Float) {
        if (count + 4 > scratch.size) return
        scratch[count++] = x
        scratch[count++] = y
        scratch[count++] = u
        scratch[count++] = v
    }

    /**
     * Adds a quad in the order top-left, top-right, bottom-right, bottom-left. `u` runs 0..1 from
     * left to right, `v` runs [vTop]..[vBottom] from the top edge to the bottom edge.
     */
    fun addQuad(
        x0: Float, y0: Float,
        x1: Float, y1: Float,
        x2: Float, y2: Float,
        x3: Float, y3: Float,
        vTop: Float, vBottom: Float,
    ) {
        add(x0, y0, 0f, vTop)
        add(x1, y1, 1f, vTop)
        add(x2, y2, 1f, vBottom)
        add(x0, y0, 0f, vTop)
        add(x2, y2, 1f, vBottom)
        add(x3, y3, 0f, vBottom)
    }

    private fun upload() {
        if (vbo == 0 || count == 0) return
        direct.clear()
        direct.put(scratch, 0, count)
        direct.flip()
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, count * 4, direct, GLES30.GL_DYNAMIC_DRAW)
    }

    fun draw() {
        if (vbo == 0 || count == 0) return
        upload()
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, 0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, 8)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, count / 4)
    }

    /** Draws with attribute 1 bound as a single float (the `v` component) for the gradient shader. */
    fun drawScalarT() {
        if (vbo == 0 || count == 0) return
        upload()
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, 0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 1, GLES30.GL_FLOAT, false, 16, 12)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, count / 4)
    }
}

/**
 * The render thread: puppet mesh + skirt, mouth layer, eye layer, recorder and snapshot.
 *
 * Per frame (passes 1..5 of the spec):
 *  1. background gradient,
 *  2. face-oval stencil write,
 *  3. puppet mesh clipped to the oval (expression warp + head pose),
 *  4. silhouette skirt (depth-tested behind the face, visible only where the face is clipped away),
 *  5. mouth layer, then eye layer, each with its own stencil clear.
 *
 * Allocation: every buffer, matrix and array used below is allocated in [onSurfaceCreated] or when
 * a new photo is loaded. `onDrawFrame` allocates nothing except the (rare) snapshot read-back.
 */
class PuppetRenderer(
    private val context: Context,
    private val state: PuppetState,
) : GLSurfaceView.Renderer {

    // ----------------------------------------------------------------------------------------- GL
    private var puppetProgram = 0
    private var layerPrograms: LayerPrograms? = null
    private var uPuppetMvp = 0
    private var uPuppetTex = 0
    private var uPuppetRim = 0

    private var restVbo = 0
    private var deltaVbo = 0
    private var indexVbo = 0
    private var indexCount = 0

    private var photoTex = 0
    private var layerTextures: LayerTextures? = null
    private var rimColor = floatArrayOf(0.35f, 0.30f, 0.28f)

    private val mouthLayer = MouthLayer()
    private val eyeLayer = EyeLayer()
    private val ovalBatch = TriBatch(FaceRig.FACE_OVAL.size * 3)
    private val backgroundBatch = QuadBatch(6)

    // ---------------------------------------------------------------------------------------- rig
    private val canonical: CanonicalMesh = loadCanonicalMesh()
    private val deformRig: DeformRig = DeformRig.build(canonical.positions)
    private var mesh: MeshData? = null
    private var loadedRig: PhotoRig? = null

    // ------------------------------------------------------------------------------------ buffers
    private val mvp = FloatArray(16)
    private val projection = FloatArray(16)
    private val modelView = FloatArray(16)
    private val delta = FloatArray((FaceFrame.LANDMARK_COUNT + FaceRig.FACE_OVAL.size) * 3)
    private var deltaBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(delta.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val ndc = FloatArray(FaceFrame.LANDMARK_COUNT * 2)
    private val residual = FloatArray(FaceFrame.LANDMARK_COUNT * 2)
    private val interpLandmarks = FloatArray(FaceFrame.LANDMARK_COUNT * 3)
    private val interpBlendshapes = FloatArray(FaceRig.BLENDSHAPE_COUNT)
    private val restBlendshapes = FloatArray(FaceRig.BLENDSHAPE_COUNT)
    private val solver = AlignmentSolver()
    private val tmpAngles = FloatArray(3)

    // ------------------------------------------------------------------- interpolation state
    private var previousFrame: FaceFrame? = null
    private var currentFrame: FaceFrame? = null
    private var blendFactor = 1f
    private var tracking = false
    private var liveAspect = 0.75f
    private var prevYaw = 0f
    private var prevPitch = 0f
    private var prevRoll = 0f
    private var curYaw = 0f
    private var curPitch = 0f
    private var curRoll = 0f

    // -------------------------------------------------------------------------- surface state
    private var surfaceWidth = 1
    private var surfaceHeight = 1
    private var fpsWindowStart = 0L
    private var fpsFrames = 0
    private var contextLost = true

    private var recorder: Recorder? = null
    private var pendingStartRecording = false

    // =============================================================================================
    // Lifecycle
    // =============================================================================================

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        Log.i(TAG, "surface created: ${GLES30.glGetString(GLES30.GL_RENDERER)}")
        GLES30.glClearColor(0.045f, 0.045f, 0.06f, 1f)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glClearStencil(0)

        puppetProgram = buildProgram(Shaders.PUPPET_VERTEX, Shaders.PUPPET_FRAGMENT)
        uPuppetMvp = GLES30.glGetUniformLocation(puppetProgram, "u_mvp")
        uPuppetTex = GLES30.glGetUniformLocation(puppetProgram, "u_tex")
        uPuppetRim = GLES30.glGetUniformLocation(puppetProgram, "u_rim")

        val solid = buildProgram(Shaders.SOLID_VERTEX, Shaders.SOLID_FRAGMENT)
        val gradient = buildProgram(Shaders.GRADIENT_VERTEX, Shaders.GRADIENT_FRAGMENT)
        val overlay = buildProgram(Shaders.OVERLAY_VERTEX, Shaders.OVERLAY_FRAGMENT)
        layerPrograms = LayerPrograms(
            solid = solid,
            gradient = gradient,
            overlay = overlay,
            uSolidZ = GLES30.glGetUniformLocation(solid, "u_z"),
            uSolidColor = GLES30.glGetUniformLocation(solid, "u_color"),
            uGradZ = GLES30.glGetUniformLocation(gradient, "u_z"),
            uGradTop = GLES30.glGetUniformLocation(gradient, "u_top"),
            uGradBottom = GLES30.glGetUniformLocation(gradient, "u_bottom"),
            uOverlayZ = GLES30.glGetUniformLocation(overlay, "u_z"),
            uOverlayTex = GLES30.glGetUniformLocation(overlay, "u_tex"),
            uOverlayTint = GLES30.glGetUniformLocation(overlay, "u_tint"),
        )

        mouthLayer.onSurfaceCreated()
        eyeLayer.onSurfaceCreated()
        ovalBatch.create()
        backgroundBatch.create()

        releaseTextures()
        photoTex = 0
        layerTextures = null
        loadedRig = null
        contextLost = false
        rebuildStaticBuffers()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceWidth = width.coerceAtLeast(1)
        surfaceHeight = height.coerceAtLeast(1)
        GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
    }

    override fun onDrawFrame(gl: GL10?) {
        if (contextLost) return
        val frameStart = System.nanoTime()

        if (state.recenterRequested.compareAndSet(true, false)) recenter()
        syncRig()
        updateInterpolation()
        updateDeltaBuffer()
        computeMatrices(surfaceWidth.toFloat() / surfaceHeight.toFloat())
        projectLandmarks()
        drawScene(surfaceWidth, surfaceHeight)

        if (pendingStartRecording) {
            pendingStartRecording = false
            startRecordingInternal()
        }
        val rec = recorder
        if (rec != null && rec.isRecording) {
            val encWidth = rec.width
            val encHeight = rec.height
            if (encWidth > 0 && encHeight > 0 && rec.beginFrame()) {
                // Re-project for the encoder aspect: the mouth/eye layers are built from NDC
                // landmarks, so they must be projected with the same matrix as the mesh.
                computeMatrices(encWidth.toFloat() / encHeight.toFloat())
                projectLandmarks()
                drawScene(encWidth, encHeight)
                rec.endFrame(System.nanoTime())
                GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
                computeMatrices(surfaceWidth.toFloat() / surfaceHeight.toFloat())
            }
        }

        if (state.snapshotRequested.compareAndSet(true, false)) captureSnapshot()

        fpsFrames++
        val now = System.nanoTime()
        if (fpsWindowStart == 0L) {
            fpsWindowStart = now
        } else if (now - fpsWindowStart >= 1_000_000_000L) {
            state.fps = fpsFrames * 1e9f / (now - fpsWindowStart)
            fpsWindowStart = now
            fpsFrames = 0
        }
        val elapsedMs = (now - frameStart) / 1_000_000
        if (elapsedMs > 60) Log.w(TAG, "slow frame: $elapsedMs ms")
    }

    // =============================================================================================
    // Recording (always invoked through GLSurfaceView.queueEvent)
    // =============================================================================================

    fun startRecording() {
        pendingStartRecording = true
    }

    fun stopRecording() {
        pendingStartRecording = false
        stopRecordingInternal()
    }

    fun shutdown() {
        stopRecordingInternal()
        releaseTextures()
    }

    private fun startRecordingInternal() {
        if (recorder?.isRecording == true) return
        val target = Recorder.chooseSize(surfaceWidth, surfaceHeight)
        val rec = Recorder(context)
        val config = state.eglConfigProvider?.invoke()
        if (rec.start(target[0], target[1], config)) {
            recorder = rec
            state.recordingStateListener?.invoke(true, null)
            Log.i(TAG, "recording ${target[0]}x${target[1]}")
        } else {
            state.recordingStateListener?.invoke(false, null)
        }
    }

    private fun stopRecordingInternal() {
        val rec = recorder ?: return
        recorder = null
        val path = rec.stop()
        state.recordingStateListener?.invoke(false, path)
        Log.i(TAG, if (path != null) "recording saved: $path" else "recording discarded")
    }

    // =============================================================================================
    // Rig loading
    // =============================================================================================

    private fun syncRig() {
        val rig = state.rig.get() ?: return
        if (rig === loadedRig) return

        val meshData = MeshBuilder.build(canonical, rig)
        mesh = meshData
        rimColor = floatArrayOf(
            (rig.rimColor[0] * RIM_DARKEN).coerceIn(0f, 1f),
            (rig.rimColor[1] * RIM_DARKEN).coerceIn(0f, 1f),
            (rig.rimColor[2] * RIM_DARKEN).coerceIn(0f, 1f),
        )

        uploadRigGeometry(meshData)
        photoTex = uploadBitmap(rig.photo, photoTex, withMipmaps = true)

        releaseTextures()
        layerTextures = LayerTextures(
            teethUpper = uploadBitmap(rig.teethUpper, 0, withMipmaps = true, anisotropy = true),
            teethLower = uploadBitmap(rig.teethLower, 0, withMipmaps = true, anisotropy = true),
            tongue = buildTongueTexture(),
            disc = buildDiscTexture(),
            irisLeft = rig.irisLeft?.let { uploadBitmap(it, 0, withMipmaps = true) } ?: 0,
            irisRight = rig.irisRight?.let { uploadBitmap(it, 0, withMipmaps = true) } ?: 0,
        )

        rig.restBlendshapes.copyInto(restBlendshapes)
        loadedRig = rig
        recenter()
        Log.i(
            TAG,
            "rig loaded: ${meshData.vertexCount} vertices, ${meshData.indices.size / 3} triangles, " +
                "teethFromPhoto=${rig.teethFromPhoto}",
        )
    }

    private fun releaseTextures() {
        val textures = layerTextures ?: return
        val names = intArrayOf(textures.teethUpper, textures.teethLower, textures.tongue, textures.disc)
        val extra = intArrayOf(textures.irisLeft, textures.irisRight)
        GLES30.glDeleteTextures(names.size, names, 0)
        GLES30.glDeleteTextures(extra.size, extra, 0)
        layerTextures = null
    }

    /** Drops all interpolation and pose history: the next frame registers from scratch. */
    private fun recenter() {
        previousFrame = null
        currentFrame = null
        blendFactor = 1f
        prevYaw = 0f
        prevPitch = 0f
        prevRoll = 0f
        curYaw = 0f
        curPitch = 0f
        curRoll = 0f
    }

    // =============================================================================================
    // Geometry
    // =============================================================================================

    private fun loadCanonicalMesh(): CanonicalMesh {
        return try {
            context.assets.open(CANONICAL_ASSET).use { stream -> MeshBuilder.loadCanonical(stream) }
        } catch (error: Throwable) {
            throw IllegalStateException(
                "Asset $CANONICAL_ASSET is missing or invalid — run `./gradlew downloadModelAssets`.",
                error,
            )
        }
    }

    private fun rebuildStaticBuffers() {
        val names = IntArray(1)
        fun release(handle: Int) {
            names[0] = handle
            GLES30.glDeleteBuffers(1, names, 0)
        }
        if (restVbo != 0) release(restVbo)
        if (deltaVbo != 0) release(deltaVbo)
        if (indexVbo != 0) release(indexVbo)
        restVbo = 0
        deltaVbo = 0
        indexVbo = 0
        mesh?.let { uploadRigGeometry(it) }
    }

    /** Uploads rest positions + UVs + skirt weight, the index list, and sizes the delta buffer. */
    private fun uploadRigGeometry(data: MeshData) {
        val interleaved = FloatArray(data.vertexCount * data.strideFloats)
        for (v in 0 until data.vertexCount) {
            val dst = v * 6
            interleaved[dst] = data.rest[v * 3]
            interleaved[dst + 1] = data.rest[v * 3 + 1]
            interleaved[dst + 2] = data.rest[v * 3 + 2]
            interleaved[dst + 3] = data.uv[v * 2]
            interleaved[dst + 4] = data.uv[v * 2 + 1]
            interleaved[dst + 5] = data.shade[v]
        }
        val vertexBuffer = ByteBuffer
            .allocateDirect(interleaved.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        vertexBuffer.put(interleaved)
        vertexBuffer.flip()

        val indexBuffer = ByteBuffer
            .allocateDirect(data.indices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asIntBuffer()
        indexBuffer.put(data.indices)
        indexBuffer.flip()

        val names = IntArray(3)
        GLES30.glGenBuffers(3, names, 0)
        restVbo = names[0]
        deltaVbo = names[1]
        indexVbo = names[2]

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, restVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, interleaved.size * 4, vertexBuffer, GLES30.GL_STATIC_DRAW)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, deltaVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.vertexCount * 3 * 4, null, GLES30.GL_DYNAMIC_DRAW)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexVbo)
        GLES30.glBufferData(
            GLES30.GL_ELEMENT_ARRAY_BUFFER,
            data.indices.size * 4,
            indexBuffer,
            GLES30.GL_STATIC_DRAW,
        )
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)

        indexCount = data.indices.size
        deltaBuffer = ByteBuffer
            .allocateDirect(data.vertexCount * 3 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    }

    // =============================================================================================
    // Per-frame animation
    // =============================================================================================

    /**
     * Picks up the newest frame and computes the interpolation factor between the previous and the
     * current inference result, so a 30 fps inference stream renders smoothly at 60 fps.
     */
    private fun updateInterpolation() {
        val frame = state.latestFrame.get()
        if (frame.hasFace && frame !== currentFrame) {
            previousFrame = currentFrame
            currentFrame = frame
            if (frame.imageWidth > 0 && frame.imageHeight > 0) {
                liveAspect = frame.imageWidth.toFloat() / frame.imageHeight
            }
            prevYaw = curYaw
            prevPitch = curPitch
            prevRoll = curRoll
            extractAngles(frame.poseMatrix, tmpAngles)
            curYaw = tmpAngles[0]
            curPitch = tmpAngles[1]
            curRoll = tmpAngles[2]
        }

        val cur = currentFrame
        val rig = loadedRig
        if (cur == null || !cur.hasFace || rig == null) {
            // No tracking yet: hold the photo's rest pose so the puppet never flickers. The
            // blendshape coefficients are zeroed rather than copied from the photo — the photo is
            // the rest pose already, and re-applying its own expression would deform it twice.
            tracking = false
            blendFactor = 1f
            rig?.restLandmarks?.copyInto(interpLandmarks)
            java.util.Arrays.fill(interpBlendshapes, 0f)
            return
        }
        tracking = true

        val prev = previousFrame
        blendFactor = if (prev == null || !prev.hasFace || cur.timestampMs <= prev.timestampMs) {
            1f
        } else {
            val nowMs = SystemClock.uptimeMillis()
            ((nowMs - prev.timestampMs).toFloat() / (cur.timestampMs - prev.timestampMs))
                .coerceIn(0f, 1f)
        }
        val t = blendFactor
        val inv = 1f - t
        if (prev == null || !prev.hasFace) {
            cur.landmarks.copyInto(interpLandmarks)
            cur.blendshapes.copyInto(interpBlendshapes)
        } else {
            for (i in interpLandmarks.indices) {
                interpLandmarks[i] = prev.landmarks[i] * inv + cur.landmarks[i] * t
            }
            for (i in 0 until FaceRig.BLENDSHAPE_COUNT) {
                interpBlendshapes[i] = prev.blendshapes[i] * inv + cur.blendshapes[i] * t
            }
        }
    }

    /**
     * `delta = deformRig(blendshapes) + damp(clamp(Umeyama(live) - photoRest))`.
     *
     * The alignment residual is damped because the blendshape rig already produces the jaw/lip
     * travel from the same coefficients; running both at full strength would double the jaw drop.
     */
    private fun updateDeltaBuffer() {
        val data = mesh ?: return
        val rig = loadedRig ?: return

        deformRig.apply(interpBlendshapes, delta)

        // The residual is only meaningful for a *live* face: at rest `interpLandmarks` is the
        // photo's own landmark set, and registering the photo against itself through a different
        // camera aspect would inject a bogus scale change into the puppet.
        val solved = tracking && solver.solve(
            live = interpLandmarks,
            rest = rig.restLandmarks,
            indices = FaceRig.ALIGN_SUBSET,
            liveAspect = liveAspect,
            restAspect = rig.aspect,
        )
        if (solved) {
            solver.residuals(
                live = interpLandmarks,
                rest = rig.restLandmarks,
                liveAspect = liveAspect,
                restAspect = rig.aspect,
                out = residual,
                clamp = RESIDUAL_CLAMP,
                toFaceUnits = 1f / rig.interOcular.coerceAtLeast(1e-4f),
                weight = RESIDUAL_WEIGHT,
                landmarkCount = data.landmarkVertexCount,
            )
            for (v in 0 until data.landmarkVertexCount) {
                delta[v * 3] += residual[v * 2]
                // MediaPipe's landmark space has y pointing down, the render space has y up.
                delta[v * 3 + 1] -= residual[v * 2 + 1]
            }
        }

        // The skirt rides along with the oval ring it was extruded from.
        for (k in data.skirtSource.indices) {
            val src = data.skirtSource[k] * 3
            val dst = data.skirtTarget[k] * 3
            delta[dst] = delta[src]
            delta[dst + 1] = delta[src + 1]
            delta[dst + 2] = delta[src + 2]
        }

        deltaBuffer.clear()
        deltaBuffer.put(delta, 0, data.vertexCount * 3)
        deltaBuffer.flip()
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, deltaVbo)
        GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, data.vertexCount * 3 * 4, deltaBuffer)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
    }

    /** Builds the model-view-projection matrix for the current pose and target aspect. */
    private fun computeMatrices(aspect: Float) {
        val t = blendFactor
        val yaw = prevYaw + (curYaw - prevYaw) * t
        val pitch = prevPitch + (curPitch - prevPitch) * t
        val roll = prevRoll + (curRoll - prevRoll) * t

        // Clamp: a single photo has no side geometry, so beyond these angles it would smear.
        val clampedYaw = yaw.coerceIn(-MAX_YAW_DEG, MAX_YAW_DEG)
        val clampedPitch = pitch.coerceIn(-MAX_PITCH_DEG, MAX_PITCH_DEG)
        val clampedRoll = roll.coerceIn(-MAX_ROLL_DEG, MAX_ROLL_DEG)

        val faceWidth = mesh?.faceWidth ?: 2.1f
        val safeAspect = aspect.coerceIn(0.35f, 3f)
        var visibleHeight = faceWidth * FACE_FILL / safeAspect
        val minHeight = faceWidth * 1.7f
        if (visibleHeight < minHeight) visibleHeight = minHeight
        val halfFovTan = tan(Math.toRadians(FOV_Y_DEG.toDouble()) * 0.5).toFloat()
        val distance = (visibleHeight * 0.5f / halfFovTan).coerceAtLeast(0.5f)

        Matrix.perspectiveM(projection, 0, FOV_Y_DEG, safeAspect, 0.1f, 200f)
        if (MIRROR_X) {
            // Selfie mirror: flip x once, here, so all landmark math stays in raw camera space.
            projection[0] = -projection[0]
            projection[4] = -projection[4]
            projection[8] = -projection[8]
            projection[12] = -projection[12]
        }

        Matrix.setIdentityM(modelView, 0)
        Matrix.translateM(modelView, 0, 0f, 0f, -distance)
        Matrix.rotateM(modelView, 0, clampedRoll, 0f, 0f, 1f)
        Matrix.rotateM(modelView, 0, clampedPitch, 1f, 0f, 0f)
        Matrix.rotateM(modelView, 0, clampedYaw, 0f, 1f, 0f)

        Matrix.multiplyMM(mvp, 0, projection, 0, modelView, 0)
    }

    /** Projects the warped rest positions with the current MVP into [ndc] (2 floats per landmark). */
    private fun projectLandmarks() {
        val data = mesh ?: return
        for (v in 0 until data.landmarkVertexCount) {
            val x = data.rest[v * 3] + delta[v * 3]
            val y = data.rest[v * 3 + 1] + delta[v * 3 + 1]
            val z = data.rest[v * 3 + 2] + delta[v * 3 + 2]
            val cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]
            val cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]
            val cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15]
            val invW = if (abs(cw) < 1e-6f) 0f else 1f / cw
            ndc[v * 2] = cx * invW
            ndc[v * 2 + 1] = cy * invW
        }
    }

    // =============================================================================================
    // Drawing
    // =============================================================================================

    private fun drawScene(width: Int, height: Int) {
        GLES30.glViewport(0, 0, width.coerceAtLeast(1), height.coerceAtLeast(1))
        GLES30.glClearStencil(0)
        GLES30.glClear(
            GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT or GLES30.GL_STENCIL_BUFFER_BIT,
        )

        drawBackground()

        val data = mesh
        val progs = layerPrograms
        val textures = layerTextures
        if (data == null || progs == null || textures == null || photoTex == 0) {
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            return
        }

        // --------------------------------------------------------------- pass 2: face-oval stencil
        ovalBatch.clear()
        val oval = FaceRig.FACE_OVAL
        var cx = 0f
        var cy = 0f
        for (index in oval) {
            cx += ndc[index * 2]
            cy += ndc[index * 2 + 1]
        }
        val invCount = 1f / oval.size
        cx *= invCount
        cy *= invCount
        for (k in oval.indices) {
            val a = oval[k]
            val b = oval[(k + 1) % oval.size]
            ovalBatch.add(cx, cy)
            ovalBatch.add(ndc[a * 2], ndc[a * 2 + 1])
            ovalBatch.add(ndc[b * 2], ndc[b * 2 + 1])
        }

        GLES30.glEnable(GLES30.GL_STENCIL_TEST)
        GLES30.glStencilMask(0xFF)
        GLES30.glStencilFunc(GLES30.GL_ALWAYS, 1, 0xFF)
        GLES30.glStencilOp(GLES30.GL_KEEP, GLES30.GL_KEEP, GLES30.GL_REPLACE)
        GLES30.glColorMask(false, false, false, false)
        GLES30.glDepthMask(false)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glUseProgram(progs.solid)
        GLES30.glUniform1f(progs.uSolidZ, LayerDepth.MASK)
        GLES30.glUniform4f(progs.uSolidColor, 0f, 0f, 0f, 1f)
        ovalBatch.draw()
        GLES30.glColorMask(true, true, true, true)

        // --------------------------------------------------------------------- pass 3: puppet mesh
        GLES30.glStencilFunc(GLES30.GL_EQUAL, 1, 0xFF)
        GLES30.glStencilOp(GLES30.GL_KEEP, GLES30.GL_KEEP, GLES30.GL_KEEP)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthFunc(GLES30.GL_LESS)
        GLES30.glDepthMask(true)

        GLES30.glUseProgram(puppetProgram)
        GLES30.glUniformMatrix4fv(uPuppetMvp, 1, false, mvp, 0)
        GLES30.glUniform1i(uPuppetTex, 0)
        GLES30.glUniform3f(uPuppetRim, rimColor[0], rimColor[1], rimColor[2])
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, photoTex)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, restVbo)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 24, 0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 24, 12)
        GLES30.glEnableVertexAttribArray(3)
        GLES30.glVertexAttribPointer(3, 1, GLES30.GL_FLOAT, false, 24, 20)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, deltaVbo)
        GLES30.glEnableVertexAttribArray(2)
        GLES30.glVertexAttribPointer(2, 3, GLES30.GL_FLOAT, false, 12, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexVbo)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_INT, 0)

        // ------------------------------------------------------------------------- pass 4: skirt
        // Stencil off, depth test on: the skirt sits behind the face, so it survives only where the
        // oval stencil clipped the face away — exactly the silhouette edge that a yaw exposes.
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_INT, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glDisableVertexAttribArray(1)
        GLES30.glDisableVertexAttribArray(2)
        GLES30.glDisableVertexAttribArray(3)

        // ----------------------------------------------------------- pass 5: mouth layer, eye layer
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glClear(GLES30.GL_STENCIL_BUFFER_BIT)
        val layerAspect = width.toFloat() / height.coerceAtLeast(1)
        mouthLayer.draw(progs, ndc, interpBlendshapes, textures, layerAspect)
        GLES30.glClear(GLES30.GL_STENCIL_BUFFER_BIT)
        eyeLayer.draw(progs, ndc, interpBlendshapes, restBlendshapes, textures, layerAspect)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glDepthMask(true)
        GLES30.glDepthFunc(GLES30.GL_LESS)
    }

    private fun drawBackground() {
        val progs = layerPrograms ?: return
        backgroundBatch.clear()
        backgroundBatch.addQuad(-1f, 1f, 1f, 1f, 1f, -1f, -1f, -1f, 0f, 1f)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(progs.gradient)
        GLES30.glUniform1f(progs.uGradZ, LayerDepth.BACKGROUND)
        GLES30.glUniform4f(progs.uGradTop, 0.085f, 0.085f, 0.105f, 1f)
        GLES30.glUniform4f(progs.uGradBottom, 0.020f, 0.020f, 0.030f, 1f)
        backgroundBatch.drawScalarT()
        GLES30.glDepthMask(true)
    }

    // =============================================================================================
    // Snapshot
    // =============================================================================================

    private fun captureSnapshot() {
        val width = surfaceWidth
        val height = surfaceHeight
        val delivery = state.snapshotDelivery ?: return
        val buffer = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer)
        buffer.rewind()
        val pixels = IntArray(width * height)
        val row = ByteArray(width * 4)
        for (y in 0 until height) {
            buffer.get(row)
            val dstRow = (height - 1 - y) * width
            var src = 0
            for (x in 0 until width) {
                val r = row[src].toInt() and 0xFF
                val g = row[src + 1].toInt() and 0xFF
                val b = row[src + 2].toInt() and 0xFF
                val a = row[src + 3].toInt() and 0xFF
                pixels[dstRow + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                src += 4
            }
        }
        delivery(pixels, width, height)
    }

    // =============================================================================================
    // Textures
    // =============================================================================================

    private fun uploadBitmap(
        bitmap: Bitmap,
        existing: Int,
        withMipmaps: Boolean,
        anisotropy: Boolean = false,
    ): Int {
        var handle = existing
        if (handle == 0) {
            val names = IntArray(1)
            GLES30.glGenTextures(1, names, 0)
            handle = names[0]
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, handle)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        if (withMipmaps) {
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER,
                GLES30.GL_LINEAR_MIPMAP_LINEAR,
            )
            if (anisotropy && supportsAnisotropy) {
                GLES30.glTexParameterf(GLES30.GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, MAX_ANISOTROPY)
            }
        } else {
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        }
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        if (withMipmaps) GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        return handle
    }

    /** Soft radial alpha disc used for the sclera, the pupil and the tongue sprite. */
    private fun buildDiscTexture(): Int {
        val size = 128
        val pixels = IntArray(size * size)
        val centre = (size - 1) * 0.5f
        for (y in 0 until size) {
            for (x in 0 until size) {
                val dx = (x - centre) / centre
                val dy = (y - centre) / centre
                val r = sqrt(dx * dx + dy * dy)
                val alpha = alphaByte((1f - r) * 6f)
                pixels[y * size + x] = (alpha shl 24) or 0x00FFFFFF
            }
        }
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, size, 0, 0, size, size)
        val handle = uploadBitmap(bitmap, 0, withMipmaps = true)
        bitmap.recycle()
        return handle
    }

    /** Procedural tongue: a warm ellipse with a darker centre crease and a soft highlight. */
    private fun buildTongueTexture(): Int {
        val width = 128
        val height = 96
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val dx = (x - width * 0.5f) / (width * 0.5f)
                val dy = (y - height * 0.5f) / (height * 0.5f)
                val r = sqrt(dx * dx + dy * dy)
                val alpha = alphaByte((1f - r) * 5f)
                val crease = (1f - abs(dx) * 3.2f).coerceIn(0f, 1f) * 0.22f
                val shade = 1f - dy * 0.18f
                val red = ((0.62f + 0.10f * shade) * 255f * (1f - crease)).toInt().coerceIn(0, 255)
                val green = ((0.22f + 0.06f * shade) * 255f * (1f - crease)).toInt().coerceIn(0, 255)
                val blue = ((0.24f + 0.06f * shade) * 255f * (1f - crease)).toInt().coerceIn(0, 255)
                pixels[y * width + x] = (alpha shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        val handle = uploadBitmap(bitmap, 0, withMipmaps = true)
        bitmap.recycle()
        return handle
    }

    private fun alphaByte(value: Float): Int = (value.coerceIn(0f, 1f) * 255f).toInt()

    // =============================================================================================
    // Program helpers + pose extraction
    // =============================================================================================

    private fun buildProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, vertexSource)
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        val status = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, status, 0)
        if (status[0] != GLES30.GL_TRUE) {
            val log = GLES30.glGetProgramInfoLog(program)
            GLES30.glDeleteProgram(program)
            throw IllegalStateException("program link failed: $log")
        }
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)
        return program
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val status = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
        if (status[0] != GLES30.GL_TRUE) {
            val log = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
            throw IllegalStateException("shader compile failed: $log")
        }
        return shader
    }

    /**
     * Extracts yaw/pitch/roll (degrees) from MediaPipe's column-major facial transformation matrix.
     *
     * The basis vectors come straight out of the matrix — column 0 is the face's right axis, column
     * 1 the up axis, column 2 the forward axis — and the angles are recovered with `atan2`, which is
     * well behaved at the moderate angles we clamp to.
     */
    private fun extractAngles(matrix: FloatArray, out: FloatArray) {
        out[0] = 0f
        out[1] = 0f
        out[2] = 0f
        if (matrix.size < 16) return
        val upX = matrix[4]
        val upY = matrix[5]
        val forwardX = matrix[8]
        val forwardY = matrix[9]
        val forwardZ = matrix[10]

        var yaw = Math.toDegrees(atan2(forwardX.toDouble(), forwardZ.toDouble())).toFloat()
        var pitch = Math.toDegrees(
            atan2(-forwardY.toDouble(), hypot(forwardX.toDouble(), forwardZ.toDouble())),
        ).toFloat()
        var roll = Math.toDegrees(atan2(-upX.toDouble(), upY.toDouble())).toFloat()
        if (!yaw.isFinite() || !pitch.isFinite() || !roll.isFinite()) {
            yaw = 0f
            pitch = 0f
            roll = 0f
        }
        // The render mirrors x (selfie mirror). Yaw and roll are rotations about y and z and flip
        // sign with that mirror; pitch is about x and does not.
        if (MIRROR_X) {
            yaw = -yaw
            roll = -roll
        }
        out[0] = yaw.coerceIn(-MAX_YAW_DEG * 1.6f, MAX_YAW_DEG * 1.6f)
        out[1] = pitch.coerceIn(-MAX_PITCH_DEG * 1.6f, MAX_PITCH_DEG * 1.6f)
        out[2] = roll.coerceIn(-MAX_ROLL_DEG * 1.6f, MAX_ROLL_DEG * 1.6f)
    }

    companion object {
        private const val TAG = "PuppetRenderer"
        private const val CANONICAL_ASSET = "canonical_face_model.obj"

        /** Selfie mirroring, applied once in the projection matrix. */
        private const val MIRROR_X = true

        private const val FOV_Y_DEG = 50f
        private const val MAX_YAW_DEG = 28f
        private const val MAX_PITCH_DEG = 20f
        private const val MAX_ROLL_DEG = 25f

        /** How much of the viewport width the head occupies. */
        private const val FACE_FILL = 1.35f

        /** Residual clamp and damping (spec: 6% of the inter-ocular distance). */
        private const val RESIDUAL_CLAMP = 0.06f
        private const val RESIDUAL_WEIGHT = 0.40f

        /** The skirt is shaded with the photo rim darkened by 45%. */
        private const val RIM_DARKEN = 0.55f

        private const val GL_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FE
        private const val MAX_ANISOTROPY = 4f

        private val supportsAnisotropy: Boolean by lazy {
            val extensions = GLES30.glGetString(GLES30.GL_EXTENSIONS) ?: ""
            extensions.contains("GL_EXT_texture_filter_anisotropic")
        }
    }
}
