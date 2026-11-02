package com.facepuppet.io

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLSurface
import android.opengl.EGLExt
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLDisplay

/**
 * MP4 recording through MediaCodec + MediaMuxer on a shared EGL context.
 *
 * How the shared-context trick works: the encoder's input `Surface` becomes a second `EGLSurface`
 * in the *same* `EGLContext` that owns the on-screen surface (that is what "shared" means here — no
 * second context, so no texture re-upload and no sharing bugs). Each recorded frame:
 *
 * ```
 * eglMakeCurrent(display, encoderSurface, encoderSurface, context)
 * renderer.drawScene(encoderWidth, encoderHeight)
 * eglPresentationTimeANDROID(display, encoderSurface, ptsNanos)
 * eglSwapBuffers(display, encoderSurface)
 * eglMakeCurrent(display, windowSurface, windowSurface, context)
 * drain encoder -> muxer
 * ```
 *
 * Everything happens on the GL thread; the renderer drives it, so the preview and the recording are
 * always the same frame.
 */
class Recorder(private val context: Context) {

    var width: Int = 0
        private set
    var height: Int = 0
        private set

    @Volatile
    var isRecording: Boolean = false
        private set

    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private val bufferInfo = MediaCodec.BufferInfo()

    private var eglDisplay: android.opengl.EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: android.opengl.EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var windowSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var inputSurface: Surface? = null

    private var trackIndex = -1
    private var muxerStarted = false
    private var sink: MediaOutput.Sink? = null
    private var firstFrameNanos = 0L
    private var framesWritten = 0
    private val stopped = AtomicBoolean(false)

    /**
     * Starts the encoder and creates the encoder-side EGL surface. Must be called with the GL
     * context current (i.e. from the GL thread).
     */
    fun start(desiredWidth: Int, desiredHeight: Int, config: EGLConfig?): Boolean {
        if (isRecording) return false
        if (config == null) {
            Log.e(TAG, "no EGLConfig available; cannot record")
            return false
        }
        return try {
            val encoder = MediaCodec.createEncoderByType(MIME_TYPE)
            val resolved = resolveSize(encoder, desiredWidth, desiredHeight)
            width = resolved[0]
            height = resolved[1]

            val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height).apply {
                setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface,
                )
                // ~5 bits per pixel is a good H.264 quality/size trade-off for 1080p captures.
                setInteger(MediaFormat.KEY_BIT_RATE, (width * height * 5).coerceIn(2_000_000, 24_000_000))
                setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val surface = encoder.createInputSurface()
            encoder.start()
            codec = encoder
            inputSurface = surface

            val created = MediaOutput.createVideoSink(context, "FacePuppet_${System.currentTimeMillis()}.mp4")
            val createdMuxer = created?.openMuxer()
            if (created == null || createdMuxer == null) {
                Log.e(TAG, "could not create output file")
                created?.discard()
                encoder.stop()
                encoder.release()
                codec = null
                return false
            }
            sink = created
            muxer = createdMuxer

            eglDisplay = EGL14.eglGetCurrentDisplay()
            eglContext = EGL14.eglGetCurrentContext()
            windowSurface = EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW)

            val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
            eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, config, surface, surfaceAttribs, 0)
            if (eglSurface == EGL14.EGL_NO_SURFACE) {
                Log.e(TAG, "eglCreateWindowSurface failed: 0x${Integer.toHexString(EGL14.eglGetError())}")
                releaseAll()
                sink?.discard()
                sink = null
                return false
            }
            firstFrameNanos = 0L
            framesWritten = 0
            stopped.set(false)
            isRecording = true
            Log.i(TAG, "encoder started: ${width}x$height @ $FRAME_RATE fps")
            true
        } catch (error: Throwable) {
            Log.e(TAG, "failed to start recorder", error)
            releaseAll()
            sink?.discard()
            sink = null
            false
        }
    }

    /** Makes the encoder surface current. Returns false if the surface is gone. */
    fun beginFrame(): Boolean {
        if (!isRecording) return false
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            Log.w(TAG, "eglMakeCurrent(encoder) failed")
            return false
        }
        return true
    }

    /** Timestamps and presents the encoder frame, restores the window surface, drains the codec. */
    fun endFrame(presentationTimeNanos: Long) {
        if (!isRecording) return
        if (firstFrameNanos == 0L) firstFrameNanos = presentationTimeNanos
        val pts = (presentationTimeNanos - firstFrameNanos).coerceAtLeast(0L)
        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, pts)
        if (!EGL14.eglSwapBuffers(eglDisplay, eglSurface)) {
            Log.w(TAG, "eglSwapBuffers(encoder) failed: 0x${Integer.toHexString(EGL14.eglGetError())}")
        }
        EGL14.eglMakeCurrent(eglDisplay, windowSurface, windowSurface, eglContext)
        framesWritten++
        drainEncoder(false)
    }

    /** Stops the encoder, finalises the MP4 and returns a human-readable location. */
    fun stop(): String? {
        if (!isRecording && codec == null) return null
        isRecording = false
        drainEncoder(true)
        val label = sink?.label
        val ok = releaseAll()
        if (framesWritten < 2 || !ok) {
            sink?.discard()
            Log.w(TAG, "recording discarded (frames=$framesWritten)")
            return null
        }
        sink?.finish()
        Log.i(TAG, "MP4 finalised: $label ($framesWritten frames)")
        return label
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val encoder = codec ?: return
        val mux = muxer
        if (endOfStream) {
            try {
                encoder.signalEndOfInputStream()
            } catch (error: Throwable) {
                Log.w(TAG, "signalEndOfInputStream failed", error)
            }
        }
        var guard = 0
        while (guard++ < 256) {
            val timeoutUs = if (endOfStream) 10_000L else 0L
            val index = try {
                encoder.dequeueOutputBuffer(bufferInfo, timeoutUs)
            } catch (error: Throwable) {
                Log.w(TAG, "dequeueOutputBuffer failed", error)
                return
            }
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) return
                    if (stopped.get()) return
                }

                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (mux != null && !muxerStarted) {
                        trackIndex = mux.addTrack(encoder.outputFormat)
                        mux.start()
                        muxerStarted = true
                    }
                }

                index >= 0 -> {
                    val buffer = encoder.getOutputBuffer(index)
                    if (buffer != null && bufferInfo.size > 0 && mux != null && muxerStarted) {
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                            buffer.position(bufferInfo.offset)
                            buffer.limit(bufferInfo.offset + bufferInfo.size)
                            try {
                                mux.writeSampleData(trackIndex, buffer, bufferInfo)
                            } catch (error: Throwable) {
                                Log.w(TAG, "writeSampleData failed", error)
                            }
                        }
                    }
                    encoder.releaseOutputBuffer(index, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        stopped.set(true)
                        return
                    }
                }
            }
        }
    }

    private fun releaseAll(): Boolean {
        var ok = true
        if (eglSurface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglMakeCurrent(eglDisplay, windowSurface, windowSurface, eglContext)
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            eglSurface = EGL14.EGL_NO_SURFACE
        }
        try {
            codec?.stop()
        } catch (error: Throwable) {
            ok = false
            Log.w(TAG, "codec stop failed", error)
        }
        try {
            codec?.release()
        } catch (error: Throwable) {
            Log.w(TAG, "codec release failed", error)
        }
        codec = null
        inputSurface?.release()
        inputSurface = null
        if (muxerStarted) {
            try {
                muxer?.stop()
            } catch (error: Throwable) {
                ok = false
                Log.w(TAG, "muxer stop failed", error)
            }
        }
        try {
            muxer?.release()
        } catch (error: Throwable) {
            Log.w(TAG, "muxer release failed", error)
        }
        muxer = null
        muxerStarted = false
        trackIndex = -1
        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        windowSurface = EGL14.EGL_NO_SURFACE
        return ok
    }

    /** Rounds the requested size to something the encoder actually supports. */
    private fun resolveSize(encoder: MediaCodec, desiredWidth: Int, desiredHeight: Int): IntArray {
        val videoCaps = try {
            encoder.codecInfo.getCapabilitiesForType(MIME_TYPE).videoCapabilities
        } catch (error: Throwable) {
            Log.w(TAG, "no video capabilities, using requested size", error)
            return intArrayOf(clamp16(desiredWidth), clamp16(desiredHeight))
        }
        val alignW = videoCaps.widthAlignment.coerceAtLeast(2)
        val alignH = videoCaps.heightAlignment.coerceAtLeast(2)
        for (scale in SCALES) {
            val w = align((desiredWidth * scale).toInt(), alignW)
            val h = align((desiredHeight * scale).toInt(), alignH)
            if (w > 0 && h > 0 && videoCaps.isSizeSupported(w, h)) return intArrayOf(w, h)
        }
        // Last resort: let the codec tell us a legal size near 720p.
        val w = clampInto(align(720, alignW), videoCaps.supportedWidths.lower, videoCaps.supportedWidths.upper)
        val h = clampInto(align(1280, alignH), videoCaps.supportedHeights.lower, videoCaps.supportedHeights.upper)
        return intArrayOf(w, h)
    }

    private fun align(value: Int, alignment: Int): Int {
        val v = if (value <= 0) alignment else value
        return ((v + alignment - 1) / alignment) * alignment
    }

    private fun clamp16(value: Int): Int = align(value, 16)

    private fun clampInto(value: Int, min: Int, max: Int): Int = value.coerceIn(min, max)

    companion object {
        private const val TAG = "Recorder"
        private const val MIME_TYPE = "video/avc"
        private const val FRAME_RATE = 30
        private val SCALES = floatArrayOf(1f, 0.75f, 0.5f, 0.34f)

        /**
         * Portrait 1080x1920 (or the closest size that matches the viewport aspect) is the target;
         * [Recorder.start] reduces the size further if the encoder does not support it.
         */
        fun chooseSize(viewWidth: Int, viewHeight: Int): IntArray {
            val aspect = if (viewHeight > 0) viewWidth.toFloat() / viewHeight else 9f / 16f
            val longEdge = 1920
            var w: Int
            var h: Int
            if (aspect <= 1f) {
                w = (longEdge * aspect).toInt()
                h = longEdge
            } else {
                w = longEdge
                h = (longEdge / aspect).toInt()
            }
            w = alignStatic(w, 16)
            h = alignStatic(h, 16)
            return intArrayOf(w, h)
        }

        private fun alignStatic(value: Int, alignment: Int): Int {
            val v = value.coerceAtLeast(alignment)
            return ((v + alignment - 1) / alignment) * alignment
        }
    }
}

/**
 * EGL config chooser that guarantees an 8-bit stencil buffer (the mouth and eye layers are stencil
 * clipped, so without a stencil buffer the layers would spill over the whole face).
 *
 * It goes through the `EGL10` handle GLSurfaceView passes in — that display is guaranteed to be
 * initialised by the time the chooser runs, whereas a fresh `EGL14.eglGetDisplay()` is not. The
 * returned object is the platform's `EGLConfigImpl`, which extends `android.opengl.EGLConfig` as
 * well, so the same config can be handed to [Recorder] for the encoder surface.
 */
class StencilConfigChooser : GLSurfaceView.EGLConfigChooser {

    /** The same config, typed for `EGL14.eglCreateWindowSurface` (null if the platform is exotic). */
    @Volatile
    var config: EGLConfig? = null
        private set

    override fun chooseConfig(egl: EGL10?, display: EGLDisplay?): javax.microedition.khronos.egl.EGLConfig {
        requireNotNull(egl) { "EGL10 handle missing" }
        requireNotNull(display) { "EGL display missing" }
        val attribs = intArrayOf(
            EGL10.EGL_RED_SIZE, 8,
            EGL10.EGL_GREEN_SIZE, 8,
            EGL10.EGL_BLUE_SIZE, 8,
            EGL10.EGL_ALPHA_SIZE, 8,
            EGL10.EGL_DEPTH_SIZE, 24,
            EGL10.EGL_STENCIL_SIZE, 8,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL10.EGL_NONE,
        )
        val configs = arrayOfNulls<javax.microedition.khronos.egl.EGLConfig>(1)
        val count = IntArray(1)
        if (!egl.eglChooseConfig(display, attribs, configs, 0, 1, count) || count[0] < 1) {
            throw IllegalArgumentException(
                "this device has no EGL config with RGBA8888 + depth24 + stencil8 + ES3",
            )
        }
        val chosen = requireNotNull(configs[0]) { "EGL returned a null config" }
        config = try {
            chosen as EGLConfig
        } catch (error: Throwable) {
            Log.w(TAG_CHOOSER, "config is not an android.opengl.EGLConfig; recording disabled", error)
            null
        }
        return chosen
    }

    private companion object {
        const val TAG_CHOOSER = "StencilConfigChooser"

        /** `EGL_RENDERABLE_TYPE`, which the legacy `EGL10` interface does not declare. */
        const val EGL_RENDERABLE_TYPE = 0x3040
        const val EGL_OPENGL_ES3_BIT_KHR = 0x0040
    }
}

/**
 * Output plumbing shared by [Recorder] and [Snapshot].
 *
 * On API 29+ everything goes through MediaStore (`Movies/FacePuppet`, `Pictures/FacePuppet`) which
 * needs no permission at all. On API 26-28 the app writes into its own external files directory —
 * still visible over ADB and in most file managers, and it keeps the manifest free of storage
 * permissions.
 */
object MediaOutput {

    abstract class Sink(val label: String) {
        /** Null for image sinks (there is no muxer involved). */
        abstract fun openMuxer(): MediaMuxer?
        open fun openStream(): OutputStream? = null
        open fun discard() {}
        open fun finish() {}
    }

    fun createVideoSink(context: Context, fileName: String): Sink? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/FacePuppet")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val resolver = context.contentResolver
            val uri: Uri? = resolver.insert(
                MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
                values,
            )
            if (uri == null) {
                null
            } else {
                val pfd = resolver.openFileDescriptor(uri, "rw")
                if (pfd == null) {
                    resolver.delete(uri, null, null)
                    null
                } else {
                    object : Sink("Movies/FacePuppet/$fileName") {
                        override fun openMuxer(): MediaMuxer =
                            MediaMuxer(pfd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

                        override fun discard() {
                            try {
                                pfd.close()
                            } catch (_: Throwable) {
                            }
                            resolver.delete(uri, null, null)
                        }

                        override fun finish() {
                            try {
                                pfd.close()
                            } catch (_: Throwable) {
                            }
                            val done = ContentValues().apply {
                                put(MediaStore.Video.Media.IS_PENDING, 0)
                            }
                            resolver.update(uri, done, null, null)
                        }
                    }
                }
            }
        } else {
            val dir = legacyDir(context, Environment.DIRECTORY_MOVIES)
            val file = File(dir, fileName)
            object : Sink(file.absolutePath) {
                override fun openMuxer(): MediaMuxer =
                    MediaMuxer(file.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

                override fun discard() {
                    file.delete()
                }
            }
        }
    } catch (error: Throwable) {
        Log.e(TAG, "could not create video sink", error)
        null
    }

    fun createImageSink(context: Context, fileName: String): Sink? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/FacePuppet")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val resolver = context.contentResolver
            val uri: Uri? = resolver.insert(
                MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
                values,
            )
            if (uri == null) {
                null
            } else {
                object : Sink("Pictures/FacePuppet/$fileName") {
                    override fun openMuxer(): MediaMuxer? = null

                    override fun openStream(): OutputStream? = resolver.openOutputStream(uri)

                    override fun discard() {
                        resolver.delete(uri, null, null)
                    }

                    override fun finish() {
                        val done = ContentValues().apply {
                            put(MediaStore.Images.Media.IS_PENDING, 0)
                        }
                        resolver.update(uri, done, null, null)
                    }
                }
            }
        } else {
            val dir = legacyDir(context, Environment.DIRECTORY_PICTURES)
            val file = File(dir, fileName)
            object : Sink(file.absolutePath) {
                override fun openMuxer(): MediaMuxer? = null

                override fun openStream(): OutputStream? = FileOutputStream(file)

                override fun discard() {
                    file.delete()
                }
            }
        }
    } catch (error: Throwable) {
        Log.e(TAG, "could not create image sink", error)
        null
    }

    private fun legacyDir(context: Context, type: String): File {
        val base = context.getExternalFilesDir(type) ?: File(context.filesDir, type)
        val dir = File(base, "FacePuppet")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    private const val TAG = "MediaOutput"

    /** Convenience for callers that only have a bitmap. */
    fun savePng(context: Context, bitmap: Bitmap, fileName: String): String? {
        val sink = createImageSink(context, fileName) ?: return null
        return try {
            val stream = sink.openStream()
            if (stream == null) {
                sink.discard()
                null
            } else {
                stream.use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                sink.finish()
                sink.label
            }
        } catch (error: Throwable) {
            Log.e(TAG, "PNG save failed", error)
            sink.discard()
            null
        }
    }
}
