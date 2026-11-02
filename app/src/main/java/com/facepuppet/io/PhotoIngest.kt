package com.facepuppet.io

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import com.facepuppet.R
import com.facepuppet.face.FaceLandmarkerEngine
import com.facepuppet.rig.FaceRig
import com.facepuppet.rig.PhotoRig
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Photo Picker ingestion: decode -> EXIF rotate -> landmark the photo -> build the [PhotoRig].
 *
 * All of this is blocking and must run off the main thread (`Dispatchers.IO`).
 *
 * The output rig is completely immutable and self-contained: the renderer only needs it to upload
 * textures and build the mesh, so the picker can be dismissed while inference finishes.
 */
object PhotoIngest {

    private const val TAG = "PhotoIngest"
    private const val MAX_EDGE = 2048
    private const val IRIS_TEXTURE_SIZE = 128
    private const val TEETH_WIDTH = 1024
    private const val TEETH_HEIGHT = 256

    sealed class Result {
        class Success(val rig: PhotoRig) : Result()
        object NoFace : Result()
        class Failure(val message: String) : Result()
    }

    fun ingest(context: Context, uri: Uri): Result {
        val photo = try {
            decodeBounded(context, uri)
        } catch (error: Throwable) {
            Log.e(TAG, "decode failed for $uri", error)
            null
        } ?: return Result.Failure("Could not read that photo.")

        val detection = FaceLandmarkerEngine.detectPhoto(context, photo) ?: return Result.NoFace

        val width = photo.width.toFloat()
        val height = photo.height.toFloat()
        val aspect = width / height
        val landmarks = detection.landmarks

        val leftX = landmarks[FaceRig.EYE_CORNER_LEFT * 3]
        val rightX = landmarks[FaceRig.EYE_CORNER_RIGHT * 3]
        val leftY = landmarks[FaceRig.EYE_CORNER_LEFT * 3 + 1]
        val rightY = landmarks[FaceRig.EYE_CORNER_RIGHT * 3 + 1]
        val eyeCornerDistancePx = hypot((leftX - rightX) * width, (leftY - rightY) * height)
        val interOcular = hypot((leftX - rightX) * aspect, leftY - rightY).coerceAtLeast(1e-3f)

        val eyeWidthPx = (eyeCornerDistancePx * 0.55f).coerceAtLeast(8f)
        val irisLeft = cropIris(photo, landmarks, FaceRig.IRIS_LEFT, eyeWidthPx)
        val irisRight = cropIris(photo, landmarks, FaceRig.IRIS_RIGHT, eyeWidthPx)

        val teeth = mouthTextures(context, photo, landmarks, eyeCornerDistancePx)

        val rig = PhotoRig(
            photo = photo,
            restLandmarks = landmarks,
            restBlendshapes = detection.blendshapes,
            irisLeft = irisLeft,
            irisRight = irisRight,
            teethUpper = teeth.upper,
            teethLower = teeth.lower,
            rimColor = rimColor(photo, landmarks),
            centerX = (leftX + rightX) * 0.5f,
            centerY = (leftY + rightY) * 0.5f,
            interOcular = interOcular,
            aspect = aspect,
            teethFromPhoto = teeth.fromPhoto,
        )
        Log.i(
            TAG,
            "photo rig: ${photo.width}x${photo.height} iod=${"%.4f".format(interOcular)} " +
                "teethFromPhoto=${rig.teethFromPhoto}",
        )
        return Result.Success(rig)
    }

    // -------------------------------------------------------------------------------------------
    // Decoding
    // -------------------------------------------------------------------------------------------

    /** Decodes to at most [MAX_EDGE] on the long edge and applies the EXIF orientation. */
    private fun decodeBounded(context: Context, uri: Uri): Bitmap? {
        val resolver = context.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sampleSize = 1
        val longEdge = max(bounds.outWidth, bounds.outHeight)
        while (longEdge / (sampleSize * 2) >= MAX_EDGE) sampleSize *= 2

        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = resolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream, null, options)
        } ?: return null

        val orientation = readExifOrientation(resolver.openInputStream(uri))
        val scaled = if (max(decoded.width, decoded.height) > MAX_EDGE) {
            val factor = MAX_EDGE.toFloat() / max(decoded.width, decoded.height)
            val w = (decoded.width * factor).toInt().coerceAtLeast(1)
            val h = (decoded.height * factor).toInt().coerceAtLeast(1)
            Bitmap.createScaledBitmap(decoded, w, h, true).also {
                if (it !== decoded) decoded.recycle()
            }
        } else {
            decoded
        }
        return applyExifOrientation(scaled, orientation)
    }

    private fun readExifOrientation(stream: java.io.InputStream?): Int {
        if (stream == null) return ExifInterface.ORIENTATION_NORMAL
        return try {
            stream.use {
                @Suppress("DEPRECATION")
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            }
        } catch (error: Throwable) {
            Log.w(TAG, "no EXIF orientation: ${error.message}")
            ExifInterface.ORIENTATION_NORMAL
        }
    }

    private fun applyExifOrientation(source: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }

            else -> return source
        }
        return try {
            val rotated = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            if (rotated !== source) source.recycle()
            rotated
        } catch (error: Throwable) {
            Log.w(TAG, "orientation failed", error)
            source
        }
    }

    // -------------------------------------------------------------------------------------------
    // Iris crops
    // -------------------------------------------------------------------------------------------

    /**
     * Circular crop of the photo around one iris, with a feathered alpha edge so it composites
     * cleanly onto the sclera.
     */
    private fun cropIris(
        photo: Bitmap,
        landmarks: FloatArray,
        irisIndex: Int,
        eyeWidthPx: Float,
    ): Bitmap? {
        val cx = landmarks[irisIndex * 3] * photo.width
        val cy = landmarks[irisIndex * 3 + 1] * photo.height
        var radius = eyeWidthPx * 0.22f
        if (!radius.isFinite() || radius < 6f) return null
        radius = radius.coerceIn(16f, 200f)

        val left = (cx - radius).coerceAtLeast(0f)
        val top = (cy - radius).coerceAtLeast(0f)
        val right = (cx + radius).coerceAtMost(photo.width.toFloat())
        val bottom = (cy + radius).coerceAtMost(photo.height.toFloat())
        if (right - left < 4f || bottom - top < 4f) return null

        val size = IRIS_TEXTURE_SIZE
        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(photo, RectF(left, top, right, bottom), RectF(0f, 0f, size.toFloat(), size.toFloat()), paint)

        val mask = Paint(Paint.ANTI_ALIAS_FLAG)
        mask.shader = RadialGradient(
            size * 0.5f,
            size * 0.5f,
            size * 0.5f,
            intArrayOf(0xFF000000.toInt(), 0xFF000000.toInt(), 0x00000000),
            floatArrayOf(0f, 0.86f, 1f),
            Shader.TileMode.CLAMP,
        )
        mask.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        canvas.drawCircle(size * 0.5f, size * 0.5f, size * 0.5f, mask)
        return output
    }

    // -------------------------------------------------------------------------------------------
    // Teeth
    // -------------------------------------------------------------------------------------------

    private class MouthTextures(
        val upper: Bitmap,
        val lower: Bitmap,
        val fromPhoto: Boolean,
    )

    /**
     * If the photo's mouth is open enough (inner-lip gap > 0.08 x inter-ocular) the real teeth are
     * cropped out of the photo, which is always better than any generated strip. Otherwise the two
     * bundled 1024x256 atlases are used.
     */
    private fun mouthTextures(
        context: Context,
        photo: Bitmap,
        landmarks: FloatArray,
        eyeCornerDistancePx: Float,
    ): MouthTextures {
        val upperLip = landmarks[FaceRig.LIP_CENTER_TOP * 3 + 1] * photo.height
        val lowerLip = landmarks[FaceRig.LIP_CENTER_BOTTOM * 3 + 1] * photo.height
        val gap = abs(lowerLip - upperLip)
        val leftCorner = landmarks[FaceRig.LIP_INNER_LEFT * 3] * photo.width
        val rightCorner = landmarks[FaceRig.LIP_INNER_RIGHT * 3] * photo.width
        val openness = gap / eyeCornerDistancePx.coerceAtLeast(1f)

        if (openness > 0.08f) {
            val mouthLeft = min(leftCorner, rightCorner)
            val mouthRight = max(leftCorner, rightCorner)
            val padX = (mouthRight - mouthLeft) * 0.04f
            val upperBand = RectF(
                mouthLeft - padX,
                (min(upperLip, lowerLip) - gap * 0.18f).coerceAtLeast(0f),
                mouthRight + padX,
                min(upperLip, lowerLip) + gap * 0.50f,
            )
            val lowerBand = RectF(
                mouthLeft - padX,
                max(upperLip, lowerLip) - gap * 0.50f,
                mouthRight + padX,
                max(upperLip, lowerLip) + gap * 0.18f,
            )
            val upper = cropRegion(photo, upperBand)
            val lower = cropRegion(photo, lowerBand)
            if (upper != null && lower != null) {
                return MouthTextures(upper, lower, fromPhoto = true)
            }
            Log.w(TAG, "mouth crop failed, using bundled teeth")
        }
        return MouthTextures(
            upper = loadDrawable(context, R.drawable.teeth_upper),
            lower = loadDrawable(context, R.drawable.teeth_lower),
            fromPhoto = false,
        )
    }

    private fun cropRegion(photo: Bitmap, region: RectF): Bitmap? {
        val left = region.left.coerceIn(0f, photo.width - 2f)
        val top = region.top.coerceIn(0f, photo.height - 2f)
        val right = region.right.coerceIn(left + 2f, photo.width.toFloat())
        val bottom = region.bottom.coerceIn(top + 2f, photo.height.toFloat())
        val output = Bitmap.createBitmap(TEETH_WIDTH, TEETH_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawBitmap(
            photo,
            RectF(left, top, right, bottom),
            RectF(0f, 0f, TEETH_WIDTH.toFloat(), TEETH_HEIGHT.toFloat()),
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG),
        )
        return output
    }

    private fun loadDrawable(context: Context, resId: Int): Bitmap {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return BitmapFactory.decodeResource(context.resources, resId, options)
    }

    // -------------------------------------------------------------------------------------------
    // Rim colour
    // -------------------------------------------------------------------------------------------

    /**
     * Average colour of the photo along the face-oval boundary, sampled slightly outside the ring so
     * the skirt picks up hair/jaw colour rather than cheek. The renderer darkens it by 45%.
     */
    private fun rimColor(photo: Bitmap, landmarks: FloatArray): FloatArray {
        var r = 0f
        var g = 0f
        var b = 0f
        var samples = 0
        val centerX = (landmarks[FaceRig.EYE_CORNER_LEFT * 3] + landmarks[FaceRig.EYE_CORNER_RIGHT * 3]) * 0.5f
        val centerY = (landmarks[FaceRig.EYE_CORNER_LEFT * 3 + 1] + landmarks[FaceRig.EYE_CORNER_RIGHT * 3 + 1]) * 0.5f
        for (index in FaceRig.FACE_OVAL) {
            val nx = landmarks[index * 3]
            val ny = landmarks[index * 3 + 1]
            val x = ((centerX + (nx - centerX) * 1.03f) * photo.width).toInt()
            val y = ((centerY + (ny - centerY) * 1.03f) * photo.height).toInt()
            if (x < 0 || y < 0 || x >= photo.width || y >= photo.height) continue
            val pixel = photo.getPixel(x, y)
            r += ((pixel shr 16) and 0xFF) / 255f
            g += ((pixel shr 8) and 0xFF) / 255f
            b += (pixel and 0xFF) / 255f
            samples++
        }
        if (samples == 0) return floatArrayOf(0.42f, 0.33f, 0.28f)
        return floatArrayOf(r / samples, g / samples, b / samples)
    }
}
