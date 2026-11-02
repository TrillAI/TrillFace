package com.facepuppet.io

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executors

/**
 * PNG snapshots.
 *
 * The pixels come straight from `glReadPixels` on the render thread (already flipped to top-down
 * order by [com.facepuppet.render.PuppetRenderer]), so this object only turns them into a bitmap and
 * writes them out — both off the GL thread, because a 1080x2400 PNG encode is tens of milliseconds
 * and would drop frames.
 */
object Snapshot {

    private const val TAG = "Snapshot"

    private val worker = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "FacePuppet-snapshot").apply { priority = Thread.MIN_PRIORITY }
    }
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * Encodes and stores the frame; [onSaved] receives the human readable location (or null on
     * failure) on the main thread.
     */
    fun save(
        context: Context,
        pixels: IntArray,
        width: Int,
        height: Int,
        onSaved: (String?) -> Unit,
    ) {
        val appContext = context.applicationContext
        worker.execute {
            val label = try {
                val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
                val name = "FacePuppet_${System.currentTimeMillis()}.png"
                val result = MediaOutput.savePng(appContext, bitmap, name)
                bitmap.recycle()
                result
            } catch (error: Throwable) {
                Log.e(TAG, "snapshot failed", error)
                null
            }
            mainHandler.post { onSaved(label) }
        }
    }
}
