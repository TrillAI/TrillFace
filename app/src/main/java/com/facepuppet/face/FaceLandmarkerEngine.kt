package com.facepuppet.face

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.facepuppet.rig.FaceRig

/**
 * Thin, allocation-conscious wrapper around MediaPipe's Face Landmarker.
 *
 * Two instances are used:
 *  * a `LIVE_STREAM` instance for the camera (GPU delegate, CPU fallback) whose result callback is
 *    invoked on MediaPipe's worker thread and immediately publishes a [FaceFrame];
 *  * short-lived `IMAGE` instances for registering an uploaded photo ([detectPhoto]).
 *
 * Model: `assets/face_landmarker.task` built with blendshapes and transformation matrices, which
 * yields 478 landmarks (468 + 2 x 5 iris refinement).
 */
class FaceLandmarkerEngine(private val context: Context) {

    /** Called on MediaPipe's worker thread with a fresh immutable frame. */
    @Volatile
    var onFrame: ((FaceFrame) -> Unit)? = null

    /** Called on MediaPipe's worker thread when the graph reports an error. */
    @Volatile
    var onError: ((String) -> Unit)? = null

    /** "GPU" or "CPU" — logged once so a field report says which path was taken. */
    @Volatile
    var delegateName: String = "none"
        private set

    private var landmarker: FaceLandmarker? = null

    /** Live-mode is single-slot: a new frame is only submitted when the previous one returned. */
    @Volatile
    private var inFlight = false

    private var lastTimestampMs = 0L
    private val landmarksScratch = FloatArray(FaceFrame.LANDMARK_COUNT * 3)
    private val blendshapesScratch = FloatArray(FaceRig.BLENDSHAPE_COUNT)
    private val poseScratch = FloatArray(16)

    fun initialize(): Boolean {
        release()
        if (startWith(Delegate.GPU)) {
            delegateName = "GPU"
        } else {
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU")
            if (!startWith(Delegate.CPU)) {
                delegateName = "none"
                onError?.invoke("Face model failed to load (CPU fallback also failed)")
                return false
            }
            delegateName = "CPU"
        }
        Log.i(TAG, "FaceLandmarker ready (delegate=$delegateName, LIVE_STREAM)")
        return true
    }

    private fun startWith(delegate: Delegate): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(delegate)
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setOutputFaceBlendshapes(true)
                .setOutputFacialTransformationMatrixes(true)
                .setResultListener { result, inputImage ->
                    handleResult(result, inputImage.width, inputImage.height)
                }
                .setErrorListener { error ->
                    inFlight = false
                    Log.e(TAG, "landmarker error: ${error.message}", error)
                    onError?.invoke(error.message ?: "landmarker error")
                }
                .build()
            landmarker = FaceLandmarker.createFromOptions(context, options)
            true
        } catch (error: Throwable) {
            // A GPU delegate failure surfaces as anything from RuntimeException to a native
            // UnsatisfiedLinkError; the CPU fallback is the documented remedy for both.
            Log.w(TAG, "delegate $delegate init failed: ${error.message}")
            landmarker?.close()
            landmarker = null
            false
        }
    }

    /**
     * Submits an **already upright** camera frame ([com.facepuppet.camera.CameraController] rotates
     * the buffer, because rotation via ImageProcessingOptions is not applied consistently for
     * LIVE_STREAM graphs and silently yields sideways landmarks on some devices).
     *
     * Returns false when the previous inference is still running or timestamps would go backwards —
     * both mean "drop this frame", which is the latest-wins policy between 30 fps inference and a
     * 60 fps render loop.
     */
    fun detectAsync(uprightImage: MPImage, timestampMs: Long): Boolean {
        val engine = landmarker ?: return false
        if (inFlight) return false
        val stamp = if (timestampMs <= lastTimestampMs) lastTimestampMs + 1 else timestampMs
        lastTimestampMs = stamp
        inFlight = true
        return try {
            engine.detectAsync(uprightImage, stamp)
            true
        } catch (error: Throwable) {
            inFlight = false
            Log.e(TAG, "detectAsync failed", error)
            false
        }
    }

    /**
     * Wraps an upright camera bitmap and submits it. The bitmap is *not* copied by
     * `BitmapImageBuilder`, so the caller must hand over a fresh bitmap per inference (which
     * [com.facepuppet.camera.CameraController] does) and must not recycle it afterwards.
     */
    fun submitBitmap(uprightBitmap: Bitmap, timestampMs: Long): Boolean {
        if (landmarker == null) return false
        return try {
            detectAsync(BitmapImageBuilder(uprightBitmap).build(), timestampMs)
        } catch (error: Throwable) {
            Log.e(TAG, "submitBitmap failed", error)
            false
        }
    }

    private fun handleResult(result: FaceLandmarkerResult, width: Int, height: Int) {
        inFlight = false
        val face = result.faceLandmarks().firstOrNull()
        if (face == null) {
            onFrame?.invoke(FaceFrame.noFace(result.timestampMs(), width, height))
            return
        }

        java.util.Arrays.fill(landmarksScratch, 0f)
        val count = minOf(face.size, FaceFrame.LANDMARK_COUNT)
        for (i in 0 until count) {
            val landmark: NormalizedLandmark = face[i]
            landmarksScratch[i * 3] = landmark.x()
            landmarksScratch[i * 3 + 1] = landmark.y()
            landmarksScratch[i * 3 + 2] = landmark.z()
        }
        if (count < FaceFrame.LANDMARK_COUNT) {
            // Without iris refinement the 10 iris points are absent. Park them between the eye
            // corners so the eye layer degrades to a centred iris instead of drawing at (0, 0).
            val eyeMidX = (landmarksScratch[FaceRig.EYE_CORNER_LEFT * 3] +
                landmarksScratch[FaceRig.EYE_CORNER_RIGHT * 3]) * 0.5f
            val eyeMidY = (landmarksScratch[FaceRig.EYE_CORNER_LEFT * 3 + 1] +
                landmarksScratch[FaceRig.EYE_CORNER_RIGHT * 3 + 1]) * 0.5f
            for (i in count until FaceFrame.LANDMARK_COUNT) {
                landmarksScratch[i * 3] = eyeMidX
                landmarksScratch[i * 3 + 1] = eyeMidY
            }
        }

        java.util.Arrays.fill(blendshapesScratch, 0f)
        val blendshapes = result.faceBlendshapes().orElse(null)
        if (!blendshapes.isNullOrEmpty()) {
            val categories: List<Category> = blendshapes[0]
            for (category in categories) {
                val name = category.categoryName()
                val index = NAME_INDEX[name]
                if (index != null) blendshapesScratch[index] = category.score()
            }
        }

        java.util.Arrays.fill(poseScratch, 0f)
        val matrices = result.facialTransformationMatrixes().orElse(null)
        if (!matrices.isNullOrEmpty()) {
            val data = matrices[0].data()
            val n = minOf(data.size, 16)
            System.arraycopy(data, 0, poseScratch, 0, n)
        }

        onFrame?.invoke(
            FaceFrame.create(
                timestampMs = result.timestampMs(),
                landmarks = landmarksScratch.copyOf(),
                blendshapes = blendshapesScratch.copyOf(),
                poseMatrix = poseScratch.copyOf(),
                imageWidth = width,
                imageHeight = height,
                hasFace = true,
            ),
        )
    }

    fun release() {
        inFlight = false
        landmarker?.close()
        landmarker = null
    }

    /** Result of registering an uploaded photo. */
    class PhotoDetection(
        val landmarks: FloatArray,
        val blendshapes: FloatArray,
        val width: Int,
        val height: Int,
    )

    companion object {
        private const val TAG = "FaceLandmarkerEngine"
        private const val MODEL_ASSET = "face_landmarker.task"

        private val NAME_INDEX: Map<String, Int> =
            FaceRig.BLENDSHAPE_NAMES.withIndex().associate { (index, name) -> name to index }

        /**
         * Runs the landmarker in IMAGE mode on a decoded photo. Blocking — call from a background
         * thread. Returns null when the photo contains no face.
         */
        fun detectPhoto(context: Context, bitmap: Bitmap): PhotoDetection? {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(Delegate.CPU)
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.4f)
                .setMinFacePresenceConfidence(0.4f)
                .setMinTrackingConfidence(0.4f)
                .setOutputFaceBlendshapes(true)
                .setOutputFacialTransformationMatrixes(true)
                .build()

            var engine: FaceLandmarker? = null
            return try {
                engine = FaceLandmarker.createFromOptions(context, options)
                val image: MPImage = BitmapImageBuilder(bitmap).build()
                val result = engine.detect(image)
                val face = result.faceLandmarks().firstOrNull() ?: return null
                val landmarks = FloatArray(FaceFrame.LANDMARK_COUNT * 3)
                val count = minOf(face.size, FaceFrame.LANDMARK_COUNT)
                for (i in 0 until count) {
                    landmarks[i * 3] = face[i].x()
                    landmarks[i * 3 + 1] = face[i].y()
                    landmarks[i * 3 + 2] = face[i].z()
                }
                if (count < FaceFrame.LANDMARK_COUNT) {
                    val midX = (landmarks[FaceRig.EYE_CORNER_LEFT * 3] +
                        landmarks[FaceRig.EYE_CORNER_RIGHT * 3]) * 0.5f
                    val midY = (landmarks[FaceRig.EYE_CORNER_LEFT * 3 + 1] +
                        landmarks[FaceRig.EYE_CORNER_RIGHT * 3 + 1]) * 0.5f
                    for (i in count until FaceFrame.LANDMARK_COUNT) {
                        landmarks[i * 3] = midX
                        landmarks[i * 3 + 1] = midY
                    }
                }

                val blendshapes = FloatArray(FaceRig.BLENDSHAPE_COUNT)
                result.faceBlendshapes().orElse(null)?.firstOrNull()?.forEach { category ->
                    NAME_INDEX[category.categoryName()]?.let { blendshapes[it] = category.score() }
                }
                PhotoDetection(landmarks, blendshapes, bitmap.width, bitmap.height)
            } catch (error: Throwable) {
                Log.e(TAG, "photo detection failed", error)
                null
            } finally {
                engine?.close()
            }
        }
    }
}
