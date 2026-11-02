package com.facepuppet.face

import com.facepuppet.rig.FaceRig

/**
 * Immutable per-inference snapshot of one detected face.
 *
 * The camera analyzer thread produces these, the MediaPipe LIVE_STREAM callback fills them in and
 * the render thread publishes the newest one through an [java.util.concurrent.atomic.AtomicReference]
 * (latest-wins, stale frames are dropped instead of queued).
 *
 * Coordinate system: MediaPipe normalised image space of the upright (rotation-corrected) camera
 * frame, i.e. x/y in [0,1] with the origin at the top-left, y pointing down, z relative to the
 * face's metric depth. Nothing in this class is mirrored — the horizontal mirror the user expects
 * from a selfie app is applied once, in the projection matrix of the renderer, so that all the
 * landmark arithmetic (registration, deformation, mouth/eye polygons) stays in raw camera space.
 *
 * Landmark budget: 478 points (468 canonical + 2 x 5 iris refinement) when the model has
 * `refine_landmarks` enabled.
 */
class FaceFrame private constructor(
    val timestampMs: Long,
    val hasFace: Boolean,
    /** `3 * LANDMARK_COUNT` floats: x, y, z per landmark, normalised image space. */
    val landmarks: FloatArray,
    /** 52 blendshape scores, indexed by [FaceRig.blendshapeIndex]. */
    val blendshapes: FloatArray,
    /** 16 floats, column-major facial transformation matrix. All zero when unavailable. */
    val poseMatrix: FloatArray,
    val imageWidth: Int,
    val imageHeight: Int,
) {

    fun landmarkX(index: Int): Float = landmarks[index * 3]

    fun landmarkY(index: Int): Float = landmarks[index * 3 + 1]

    fun landmarkZ(index: Int): Float = landmarks[index * 3 + 2]

    fun blendshape(name: String): Float = blendshapes[FaceRig.blendshapeIndex(name)]

    companion object {
        const val LANDMARK_COUNT = 478

        /** Placeholder used before the first inference result arrives. */
        fun empty(): FaceFrame = FaceFrame(
            timestampMs = 0L,
            hasFace = false,
            landmarks = FloatArray(LANDMARK_COUNT * 3),
            blendshapes = FloatArray(FaceRig.BLENDSHAPE_COUNT),
            poseMatrix = FloatArray(16),
            imageWidth = 0,
            imageHeight = 0,
        )

        /**
         * Copies the raw MediaPipe output into a new immutable frame.
         *
         * This is the only allocation on the inference path (one small object + three arrays per
         * face per inference tick). Allocation is forbidden in `onDrawFrame`, not here: the render
         * thread reads these arrays through a published reference and never mutates them.
         */
        fun create(
            timestampMs: Long,
            landmarks: FloatArray,
            blendshapes: FloatArray,
            poseMatrix: FloatArray,
            imageWidth: Int,
            imageHeight: Int,
            hasFace: Boolean = true,
        ): FaceFrame = FaceFrame(
            timestampMs = timestampMs,
            hasFace = hasFace,
            landmarks = landmarks,
            blendshapes = blendshapes,
            poseMatrix = poseMatrix,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
        )

        /** No-face frame that keeps the timestamp moving so interpolation degrades gracefully. */
        fun noFace(timestampMs: Long, imageWidth: Int, imageHeight: Int): FaceFrame = FaceFrame(
            timestampMs = timestampMs,
            hasFace = false,
            landmarks = FloatArray(LANDMARK_COUNT * 3),
            blendshapes = FloatArray(FaceRig.BLENDSHAPE_COUNT),
            poseMatrix = FloatArray(16),
            imageWidth = imageWidth,
            imageHeight = imageHeight,
        )
    }
}
