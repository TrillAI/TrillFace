package com.facepuppet.rig

import android.graphics.Bitmap

/**
 * The rig: landmark index groups, the blendshape vocabulary, the photo-side rest pose and the
 * pre-computed sparse blendshape -> vertex-offset matrix.
 *
 * Coordinate conventions used by every file in this project
 * --------------------------------------------------------
 * * MediaPipe output: normalised image space, x/y in [0,1], origin top-left, **y points down**.
 * * "Face units": origin at the centre of the photo face, +x right, **+y up**, +z towards the
 *   viewer, **inter-ocular distance (IOD) == 1.0**. All rig constants below are in face units.
 * * NDC: what the layers draw in after the renderer projects warped vertices with the puppet MVP.
 */
object FaceRig {

    // -----------------------------------------------------------------------------------------
    // Landmark index groups (478-point model, iris refinement on).
    // -----------------------------------------------------------------------------------------

    /** Outer lip contour, clockwise from the left corner (11 points). */
    val LIPS_OUTER = intArrayOf(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)

    /** Inner lip contour, clockwise from the left inner corner (12 points). */
    val LIPS_INNER = intArrayOf(78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191)

    /** Upper eyelid ring, outer corner -> inner corner (image-left eye). */
    val LEFT_EYE_UPPER = intArrayOf(33, 246, 161, 160, 159, 158, 157, 173, 133)

    /** Lower eyelid ring, outer corner -> inner corner (image-left eye). */
    val LEFT_EYE_LOWER = intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133)

    /** Upper eyelid ring, inner corner -> outer corner (image-right eye). */
    val RIGHT_EYE_UPPER = intArrayOf(362, 398, 384, 385, 386, 387, 388, 466, 263)

    /** Lower eyelid ring, inner corner -> outer corner (image-right eye). */
    val RIGHT_EYE_LOWER = intArrayOf(362, 382, 381, 380, 374, 373, 390, 249, 263)

    const val IRIS_LEFT = 468
    const val IRIS_RIGHT = 473
    const val NOSE_TIP = 1

    /** 36-point face oval, clockwise from the top of the forehead. */
    val FACE_OVAL = intArrayOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    )

    /**
     * Stable subset used to solve the photo<-live similarity transform: nose bridge plus the two
     * inner eye corners and the nose tip. These six points are the least affected by expression,
     * which is what makes the residual a usable mimic signal.
     */
    val ALIGN_SUBSET = intArrayOf(6, 9, 168, 197, 195, 133, 362, 1)

    /** Inner eye corners; their distance defines the inter-ocular distance. */
    const val EYE_CORNER_LEFT = 133
    const val EYE_CORNER_RIGHT = 362

    // Mouth anchors for the teeth rows.
    const val LIP_CENTER_TOP = 13
    const val LIP_CENTER_BOTTOM = 14
    const val LIP_INNER_LEFT = 78
    const val LIP_INNER_RIGHT = 191
    const val LIP_OUTER_BOTTOM = 17

    // Eyebrows (the inner/outer split is computed geometrically at rig-build time, so the groups
    // only need to list the brow vertices themselves).
    val LEFT_BROW = intArrayOf(70, 63, 105, 66, 107)
    val RIGHT_BROW = intArrayOf(296, 334, 293, 300, 285)

    /** Upper eyelids only: the corners stay put while the lid slides down on a blink. */
    val LEFT_UPPER_LID = intArrayOf(246, 161, 160, 159, 158, 157, 173)
    val RIGHT_UPPER_LID = intArrayOf(398, 384, 385, 386, 387, 388, 466)

    /** Cheek ring used by cheekSquint / cheekPuff. */
    val CHEEK_RING = intArrayOf(50, 101, 36, 205, 266, 280, 330, 425)

    // -----------------------------------------------------------------------------------------
    // Blendshapes
    // -----------------------------------------------------------------------------------------

    /**
     * The 52 MediaPipe face blendshapes in canonical (ARKit) order. Scores arrive from MediaPipe
     * as an unordered name/score list, so [FaceLandmarkerEngine][com.facepuppet.face.FaceLandmarkerEngine]
     * maps them by *name* into this order — no index guessing, and a future re-ordering of the
     * model cannot silently rotate the animation.
     */
    val BLENDSHAPE_NAMES = arrayOf(
        "_neutral",
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    )

    const val BLENDSHAPE_COUNT = 52

    const val B_NEUTRAL = 0
    const val B_BROW_DOWN_LEFT = 1
    const val B_BROW_DOWN_RIGHT = 2
    const val B_BROW_INNER_UP = 3
    const val B_BROW_OUTER_UP_LEFT = 4
    const val B_BROW_OUTER_UP_RIGHT = 5
    const val B_CHEEK_PUFF = 6
    const val B_CHEEK_SQUINT_LEFT = 7
    const val B_CHEEK_SQUINT_RIGHT = 8
    const val B_EYE_BLINK_LEFT = 9
    const val B_EYE_BLINK_RIGHT = 10
    const val B_EYE_LOOK_DOWN_LEFT = 11
    const val B_EYE_LOOK_DOWN_RIGHT = 12
    const val B_EYE_LOOK_IN_LEFT = 13
    const val B_EYE_LOOK_IN_RIGHT = 14
    const val B_EYE_LOOK_OUT_LEFT = 15
    const val B_EYE_LOOK_OUT_RIGHT = 16
    const val B_EYE_LOOK_UP_LEFT = 17
    const val B_EYE_LOOK_UP_RIGHT = 18
    const val B_EYE_WIDE_LEFT = 21
    const val B_EYE_WIDE_RIGHT = 22
    const val B_JAW_OPEN = 25
    const val B_MOUTH_FUNNEL = 32
    const val B_MOUTH_PUCKER = 38
    const val B_MOUTH_SMILE_LEFT = 44
    const val B_MOUTH_SMILE_RIGHT = 45
    const val B_MOUTH_STRETCH_LEFT = 46
    const val B_MOUTH_STRETCH_RIGHT = 47

    private val NAME_TO_INDEX: Map<String, Int> =
        BLENDSHAPE_NAMES.withIndex().associate { (index, name) -> name to index }

    /** Index of [name] in [BLENDSHAPE_NAMES]; `-_neutral` style aliases are tolerated. */
    fun blendshapeIndex(name: String): Int {
        NAME_TO_INDEX[name]?.let { return it }
        val stripped = name.removePrefix("_")
        return NAME_TO_INDEX[stripped] ?: NAME_TO_INDEX["_$stripped"] ?: B_NEUTRAL
    }

    fun namesToCoefficients(name: String, score: Float, out: FloatArray): Boolean {
        val index = NAME_TO_INDEX[name] ?: return false
        out[index] = score
        return true
    }
}

/**
 * Everything the renderer needs to puppet a specific photo. Built off the render thread by
 * [com.facepuppet.io.PhotoIngest], published through an AtomicReference and consumed (uploaded as
 * GL textures / VBOs) once on the render thread. Treat every field as read-only.
 */
class PhotoRig(
    /** The photo, EXIF-rotated, decoded to at most 2048 px on the long edge. */
    val photo: Bitmap,
    /** 478 x 3 normalised landmark positions of the photo's face (x/y in [0,1] of [photo]). */
    val restLandmarks: FloatArray,
    /** Blendshape scores of the photo itself; used as the blink/expression baseline. */
    val restBlendshapes: FloatArray,
    /** Square iris crops from the photo, already circular-masked, or null when not reliable. */
    val irisLeft: Bitmap?,
    val irisRight: Bitmap?,
    /** Teeth atlases: photo crops when the photo's mouth is open, otherwise the bundled strips. */
    val teethUpper: Bitmap,
    val teethLower: Bitmap,
    /** Rim colour (linear-ish 0..1 RGB) sampled from the photo's face-oval boundary. */
    val rimColor: FloatArray,
    /** Face centre in normalised photo space. */
    val centerX: Float,
    val centerY: Float,
    /** Inter-ocular distance in aspect-corrected normalised units (x * width / height). */
    val interOcular: Float,
    /** photo.width / photo.height. */
    val aspect: Float,
    /** True when the teeth atlases came from the photo itself rather than the bundled strips. */
    val teethFromPhoto: Boolean,
)

/**
 * Sparse blendshape -> vertex-offset matrix, `vertexIndex -> list of (blendshapeIndex, dx, dy)`.
 *
 * Built once per loaded photo (it depends only on the canonical mesh and the landmark constants,
 * but keeping one instance next to the rig makes ownership obvious). Applying it is a single
 * linear pass over all entries with no allocation:
 *
 * ```
 * delta.fill(0f)
 * for (e in entries) {
 *     val c = coefficients[blendIndex[e]]
 *     delta[vertex * 3]     += c * dx[e]
 *     delta[vertex * 3 + 1] += c * dy[e]
 * }
 * ```
 */
class DeformRig private constructor(
    private val vertexIndex: IntArray,
    private val blendIndex: IntArray,
    private val dx: FloatArray,
    private val dy: FloatArray,
) {

    val entryCount: Int get() = vertexIndex.size

    /**
     * Writes the summed offsets into [delta] (`3 * vertexCount` floats, must be zero-filled or
     * will be cleared here). No allocation: [delta] is the renderer's persisted buffer.
     */
    fun apply(coefficients: FloatArray, delta: FloatArray) {
        java.util.Arrays.fill(delta, 0f)
        val vi = vertexIndex
        val bi = blendIndex
        for (e in vi.indices) {
            val c = coefficients[bi[e]]
            if (c == 0f) continue
            val v = vi[e] * 3
            delta[v] += c * dx[e]
            delta[v + 1] += c * dy[e]
        }
    }

    companion object {
        /** Chin drop for a fully open jaw, in face units (IOD == 1). */
        private const val JAW_DROP = 0.42f
        private const val SMILE_UP = 0.10f
        private const val SMILE_OUT = 0.06f
        private const val PUCKER_CONTRACT = 0.10f
        private const val FUNNEL_CONTRACT = 0.07f
        private const val BROW_INNER_UP = 0.055f
        private const val BROW_OUTER_UP = 0.050f
        private const val BROW_DOWN = 0.050f
        private const val CHEEK_SQUINT = 0.020f
        private const val CHEEK_PUFF = 0.030f
        private const val LID_DOWN = 0.060f
        private const val LID_UP = 0.045f

        private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
            if (edge1 <= edge0) return if (x >= edge1) 1f else 0f
            val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        private fun clamp01(v: Float) = if (v < 0f) 0f else if (v > 1f) 1f else v

        /**
         * @param canonical 468 x 3 canonical mesh positions normalised to face units
         *        (origin at the face centre, +y up, IOD == 1). This is the *canonical* mesh, not
         *        the photo mesh: the rig must be independent of which photo is loaded.
         */
        fun build(canonical: FloatArray): DeformRig {
            val verts = canonical.size / 3
            val vList = ArrayList<Int>(512)
            val bList = ArrayList<Int>(512)
            val dxList = ArrayList<Float>(512)
            val dyList = ArrayList<Float>(512)

            fun add(vertex: Int, blend: Int, ddx: Float, ddy: Float) {
                if (vertex < 0 || vertex >= verts) return
                if (ddx == 0f && ddy == 0f) return
                vList.add(vertex)
                bList.add(blend)
                dxList.add(ddx)
                dyList.add(ddy)
            }

            fun x(v: Int) = canonical[v * 3]
            fun y(v: Int) = canonical[v * 3 + 1]

            val mouthY = (y(FaceRig.LIP_CENTER_TOP) + y(FaceRig.LIP_CENTER_BOTTOM)) * 0.5f
            val chinY = y(152)
            val jawSpan = (mouthY - chinY).coerceAtLeast(0.05f)
            val ovalHalfWidth = maxOf(
                kotlin.math.abs(x(234)),
                kotlin.math.abs(x(454)),
                0.5f,
            )

            // --- jawOpen: everything below the mouth line slides down along -y, weighted by the
            // vertical distance below the mouth so the upper face stays pinned.
            for (v in 0 until verts) {
                val below = (mouthY - y(v)) / jawSpan
                if (below <= 0f) continue
                val w = smoothstep(0f, 1f, below)
                if (w <= 0.001f) continue
                add(v, FaceRig.B_JAW_OPEN, 0f, -JAW_DROP * w)
            }

            // --- smiles: outer lip pulled up and out, strongest at the corners.
            val mouthHalfWidth = kotlin.math.abs(x(FaceRig.LIPS_OUTER[0])).coerceAtLeast(0.2f)
            for (v in FaceRig.LIPS_OUTER) {
                val w = smoothstep(0.15f, 1f, kotlin.math.abs(x(v)) / mouthHalfWidth)
                if (w <= 0.001f) continue
                if (x(v) <= 0f) {
                    add(v, FaceRig.B_MOUTH_SMILE_LEFT, -SMILE_OUT * w, SMILE_UP * w)
                    add(v, FaceRig.B_MOUTH_STRETCH_LEFT, -SMILE_OUT * 0.6f * w, 0f)
                } else {
                    add(v, FaceRig.B_MOUTH_SMILE_RIGHT, SMILE_OUT * w, SMILE_UP * w)
                    add(v, FaceRig.B_MOUTH_STRETCH_RIGHT, SMILE_OUT * 0.6f * w, 0f)
                }
            }

            // --- pucker / funnel: the whole lip ring contracts towards the mouth centre.
            val lipCenterY = mouthY
            val lipVerts = FaceRig.LIPS_OUTER + FaceRig.LIPS_INNER
            for (v in lipVerts) {
                val ox = x(v)
                val oy = y(v) - lipCenterY
                val len = kotlin.math.sqrt(ox * ox + oy * oy).coerceAtLeast(1e-4f)
                val nx = ox / len
                val ny = oy / len
                add(v, FaceRig.B_MOUTH_PUCKER, -nx * PUCKER_CONTRACT, -ny * PUCKER_CONTRACT)
                add(v, FaceRig.B_MOUTH_FUNNEL, -nx * FUNNEL_CONTRACT, -ny * FUNNEL_CONTRACT)
            }

            // --- brows. The inner/outer split is geometric (distance from the face midline), so it
            // does not depend on getting a per-index left/right convention right.
            val browVerts = FaceRig.LEFT_BROW + FaceRig.RIGHT_BROW
            var minAbsX = Float.MAX_VALUE
            var maxAbsX = 0f
            for (v in browVerts) {
                val ax = kotlin.math.abs(x(v))
                if (ax < minAbsX) minAbsX = ax
                if (ax > maxAbsX) maxAbsX = ax
            }
            if (maxAbsX <= minAbsX) maxAbsX = minAbsX + 0.1f
            for (v in browVerts) {
                val innerW = 1f - smoothstep(minAbsX, maxAbsX, kotlin.math.abs(x(v)))
                val outerW = 1f - innerW
                add(v, FaceRig.B_BROW_INNER_UP, 0f, BROW_INNER_UP * innerW)
                if (x(v) <= 0f) {
                    add(v, FaceRig.B_BROW_OUTER_UP_LEFT, 0f, BROW_OUTER_UP * outerW)
                    add(v, FaceRig.B_BROW_DOWN_LEFT, 0f, -BROW_DOWN)
                } else {
                    add(v, FaceRig.B_BROW_OUTER_UP_RIGHT, 0f, BROW_OUTER_UP * outerW)
                    add(v, FaceRig.B_BROW_DOWN_RIGHT, 0f, -BROW_DOWN)
                }
            }

            // --- cheeks: radial offsets about the face centre.
            for (v in FaceRig.CHEEK_RING) {
                val ox = x(v)
                val oy = y(v)
                val len = kotlin.math.sqrt(ox * ox + oy * oy).coerceAtLeast(1e-4f)
                val nx = ox / len
                val ny = oy / len
                val squint = if (x(v) <= 0f) FaceRig.B_CHEEK_SQUINT_LEFT else FaceRig.B_CHEEK_SQUINT_RIGHT
                add(v, squint, -nx * CHEEK_SQUINT, -ny * CHEEK_SQUINT * 0.5f)
                add(v, FaceRig.B_CHEEK_PUFF, nx * CHEEK_PUFF, ny * CHEEK_PUFF)
            }

            // --- eyelids.
            for (v in FaceRig.LEFT_UPPER_LID) {
                add(v, FaceRig.B_EYE_BLINK_LEFT, 0f, -LID_DOWN)
                add(v, FaceRig.B_EYE_WIDE_LEFT, 0f, LID_UP)
            }
            for (v in FaceRig.RIGHT_UPPER_LID) {
                add(v, FaceRig.B_EYE_BLINK_RIGHT, 0f, -LID_DOWN)
                add(v, FaceRig.B_EYE_WIDE_RIGHT, 0f, LID_UP)
            }

            // Sanity: the oval half width is only used to keep brow weights sane on odd meshes.
            if (ovalHalfWidth <= 0f) throw IllegalStateException("degenerate canonical mesh")

            val n = vList.size
            val vArr = IntArray(n)
            val bArr = IntArray(n)
            val dxArr = FloatArray(n)
            val dyArr = FloatArray(n)
            for (i in 0 until n) {
                vArr[i] = vList[i]
                bArr[i] = bList[i]
                dxArr[i] = dxList[i]
                dyArr[i] = dyList[i]
            }
            return DeformRig(vArr, bArr, dxArr, dyArr)
        }
    }
}
