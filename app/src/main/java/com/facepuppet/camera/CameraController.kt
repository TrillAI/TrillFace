package com.facepuppet.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import java.nio.ByteBuffer
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Front-camera capture via CameraX `ImageAnalysis`.
 *
 * Design notes
 * ------------
 * * Only an `ImageAnalysis` use case is bound. The visible preview is the OpenGL puppet, so a
 *   second `Preview` use case would cost a surface and a buffer queue for pixels nobody shows.
 * * `OUTPUT_IMAGE_FORMAT_RGBA_8888` keeps the conversion out of our code: the camera pipeline
 *   produces packed RGBA, we only have to strip the row padding and rotate.
 * * Frames are handed to the caller **already upright**. MediaPipe's `ImageProcessingOptions`
 *   rotation is unreliable for LIVE_STREAM graphs on some devices (landmarks come back rotated by
 *   90 degrees), and rotating here is one native `Bitmap.createBitmap` call.
 * * One `Bitmap` is allocated per inference tick on purpose: `BitmapImageBuilder` wraps the pixels
 *   without copying, so the buffer has to stay alive for the whole async inference. Recycling a
 *   shared bitmap would race the graph. The allocation is a single young-gen 640x480 buffer, and it
 *   happens off both the render thread and the UI thread.
 */
class CameraController(
    private val context: Context,
    /** Invoked on the analysis thread with an upright RGBA bitmap plus its monotonic timestamp. */
    private val onUprightFrame: (Bitmap, Long) -> Unit,
    private val onError: (String) -> Unit,
) {

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "FacePuppet-analyzer").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val mainExecutor = Executor { runnable ->
        Handler(Looper.getMainLooper()).post(runnable)
    }

    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null

    /** Reused staging buffer for the row-by-row de-padding copy (analysis thread only). */
    private var scratch: ByteBuffer? = null
    private var scratchCapacity = 0

    @Volatile
    var isRunning: Boolean = false
        private set

    @Volatile
    private var stopped = true

    fun start(lifecycleOwner: LifecycleOwner) {
        if (isRunning) return
        stopped = false
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (stopped) return@addListener
            try {
                val cameraProvider = future.get()
                provider = cameraProvider

                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetResolution(Size(TARGET_WIDTH, TARGET_HEIGHT))
                    .build()

                analyzer.setAnalyzer(analysisExecutor) { image -> analyzeFrame(image) }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    analyzer,
                )
                analysis = analyzer
                isRunning = true
                Log.i(TAG, "front camera bound at ~${TARGET_WIDTH}x$TARGET_HEIGHT RGBA")
            } catch (error: Throwable) {
                Log.e(TAG, "camera bind failed", error)
                isRunning = false
                onError("Camera unavailable: ${error.message}")
            }
        }, mainExecutor)
    }

    fun stop() {
        stopped = true
        isRunning = false
        analysis?.clearAnalyzer()
        analysis = null
        try {
            provider?.unbindAll()
        } catch (error: Throwable) {
            Log.w(TAG, "unbind failed", error)
        }
        provider = null
        // Wait for the frame that may still be inside the analyzer, so the caller can safely tear
        // the landmarker down right after stop() returns.
        try {
            analysisExecutor.submit { }.get(500, TimeUnit.MILLISECONDS)
        } catch (error: Throwable) {
            Log.w(TAG, "could not drain analyzer: ${error.message}")
        }
    }

    /** Releases the analyzer thread; call from `onDestroy`. */
    fun shutdown() {
        stop()
        analysisExecutor.shutdown()
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            if (stopped) return
            val bitmap = toUprightBitmap(image) ?: return
            onUprightFrame(bitmap, image.imageInfo.timestamp / 1_000_000L)
        } catch (error: Throwable) {
            Log.w(TAG, "frame conversion failed", error)
        } finally {
            image.close()
        }
    }

    /**
     * RGBA_8888 `ImageProxy` -> upright [Bitmap]. The plane is padded to a hardware-friendly row
     * stride, so rows are copied individually into a contiguous staging buffer.
     */
    private fun toUprightBitmap(image: ImageProxy): Bitmap? {
        val plane = image.planes.firstOrNull() ?: return null
        val width = image.width
        val height = image.height
        if (width <= 0 || height <= 0) return null

        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride.coerceAtLeast(4)
        val rowBytes = width * pixelStride
        val needed = rowBytes * height

        if (scratch == null || scratchCapacity < needed) {
            scratch = ByteBuffer.allocateDirect(needed)
            scratchCapacity = needed
        }
        val staging = scratch!!

        staging.clear()
        buffer.rewind()
        val limit = buffer.limit()
        if (rowStride == rowBytes && pixelStride == 4) {
            val end = minOf(limit, needed)
            buffer.limit(end)
            staging.put(buffer)
            buffer.limit(limit)
        } else {
            for (row in 0 until height) {
                val start = row * rowStride
                if (start >= limit) break
                val end = minOf(start + rowBytes, limit)
                buffer.limit(end)
                buffer.position(start)
                staging.put(buffer)
            }
            buffer.limit(limit)
        }
        staging.rewind()

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(staging)

        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    companion object {
        private const val TAG = "CameraController"

        /**
         * 640x480 is the sweet spot for the landmarker: ~10 ms per inference with the GPU delegate
         * on a mid-range phone, and enough resolution for the 478-point model to be stable.
         */
        const val TARGET_WIDTH = 640
        const val TARGET_HEIGHT = 480
    }
}
