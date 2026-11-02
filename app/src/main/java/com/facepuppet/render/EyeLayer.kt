package com.facepuppet.render

import android.opengl.GLES30
import com.facepuppet.rig.FaceRig
import kotlin.math.abs
import kotlin.math.max

/**
 * The eyes: eyelid-shaped stencil opening, sclera, iris patch cropped from the photo, pupil disc.
 *
 * The opening polygon is the eyelid ring squashed vertically by `1 - 0.9 * eyeBlink` about the eye
 * centre, so a blink closes the visible slit while the painted lid (which is part of the puppet
 * mesh) keeps moving with the mesh deformation.
 *
 * Compositing is *conditional*: when the photo and the live face agree on the blink state and the
 * gaze is centred, the layer is skipped entirely and the photo's own painted eye shows through,
 * which avoids resampling artefacts at rest.
 */
class EyeLayer {

    /** 2 eyes x (upper 9 + lower 7) points, fanned from the centroid. */
    private val maskBatch = TriBatch(2 * 16 * 3)
    private val scleraBatch = QuadBatch(12)
    private val irisLeftBatch = QuadBatch(6)
    private val irisRightBatch = QuadBatch(6)
    private val pupilBatch = QuadBatch(12)

    fun onSurfaceCreated() {
        maskBatch.create()
        scleraBatch.create()
        irisLeftBatch.create()
        irisRightBatch.create()
        pupilBatch.create()
    }

    fun release() {
        maskBatch.destroy()
        scleraBatch.destroy()
        irisLeftBatch.destroy()
        irisRightBatch.destroy()
        pupilBatch.destroy()
    }

    /**
     * @param ndc projected landmark positions of the deformed + posed puppet.
     * @param coef live blendshape coefficients.
     * @param restCoef blendshape coefficients measured on the photo itself.
     */
    fun draw(
        progs: LayerPrograms,
        ndc: FloatArray,
        coef: FloatArray,
        restCoef: FloatArray,
        textures: LayerTextures,
        aspect: Float,
    ) {
        // NDC x and y are not isotropic: 1 unit of x is `aspect` times shorter than 1 unit of y.
        // Iris/pupil sprites must be square *in pixels*, so their radius is derived in y units and
        // divided back out when written as an x extent.
        val a = aspect.coerceIn(0.2f, 5f)
        val leftBlink = blend(coef, restCoef, FaceRig.B_EYE_BLINK_LEFT)
        val rightBlink = blend(coef, restCoef, FaceRig.B_EYE_BLINK_RIGHT)
        val leftWide = coef[FaceRig.B_EYE_WIDE_LEFT]
        val rightWide = coef[FaceRig.B_EYE_WIDE_RIGHT]

        val leftLookX = coef[FaceRig.B_EYE_LOOK_IN_LEFT] - coef[FaceRig.B_EYE_LOOK_OUT_LEFT]
        val leftLookY = coef[FaceRig.B_EYE_LOOK_UP_LEFT] - coef[FaceRig.B_EYE_LOOK_DOWN_LEFT]
        val rightLookX = coef[FaceRig.B_EYE_LOOK_IN_RIGHT] - coef[FaceRig.B_EYE_LOOK_OUT_RIGHT]
        val rightLookY = coef[FaceRig.B_EYE_LOOK_UP_RIGHT] - coef[FaceRig.B_EYE_LOOK_DOWN_RIGHT]

        val leftActive = shouldComposite(
            coef[FaceRig.B_EYE_BLINK_LEFT], restCoef[FaceRig.B_EYE_BLINK_LEFT],
            coef[FaceRig.B_EYE_LOOK_IN_LEFT] + coef[FaceRig.B_EYE_LOOK_OUT_LEFT] +
                coef[FaceRig.B_EYE_LOOK_UP_LEFT] + coef[FaceRig.B_EYE_LOOK_DOWN_LEFT],
        )
        val rightActive = shouldComposite(
            coef[FaceRig.B_EYE_BLINK_RIGHT], restCoef[FaceRig.B_EYE_BLINK_RIGHT],
            coef[FaceRig.B_EYE_LOOK_IN_RIGHT] + coef[FaceRig.B_EYE_LOOK_OUT_RIGHT] +
                coef[FaceRig.B_EYE_LOOK_UP_RIGHT] + coef[FaceRig.B_EYE_LOOK_DOWN_RIGHT],
        )
        if (!leftActive && !rightActive) return

        // Blink is a *relative* signal: if the photo already shows a half-closed eye, an identical
        // live blink must not close it a second time, which is what `blend` takes care of.
        val leftLid = (1f - 0.9f * leftBlink + 0.35f * leftWide).coerceIn(0f, 1.35f)
        val rightLid = (1f - 0.9f * rightBlink + 0.35f * rightWide).coerceIn(0f, 1.35f)

        maskBatch.clear()
        var leftCenterX = 0f
        var leftCenterY = 0f
        var leftHalfWidth = 0f
        var leftHalfHeight = 0f
        var rightCenterX = 0f
        var rightCenterY = 0f
        var rightHalfWidth = 0f
        var rightHalfHeight = 0f

        if (leftActive) {
            val g = buildOpening(
                ndc,
                FaceRig.LEFT_EYE_UPPER,
                FaceRig.LEFT_EYE_LOWER,
                leftLid,
            )
            leftCenterX = g.centerX
            leftCenterY = g.centerY
            leftHalfWidth = g.halfWidth
            leftHalfHeight = g.halfHeight
            addMaskFan(g)
        }
        if (rightActive) {
            val g = buildOpening(
                ndc,
                FaceRig.RIGHT_EYE_UPPER,
                FaceRig.RIGHT_EYE_LOWER,
                rightLid,
            )
            rightCenterX = g.centerX
            rightCenterY = g.centerY
            rightHalfWidth = g.halfWidth
            rightHalfHeight = g.halfHeight
            addMaskFan(g)
        }

        // ------------------------------------------------------------------------------ stencil
        GLES30.glEnable(GLES30.GL_STENCIL_TEST)
        GLES30.glStencilMask(0xFF)
        GLES30.glStencilFunc(GLES30.GL_ALWAYS, 2, 0xFF)
        GLES30.glStencilOp(GLES30.GL_KEEP, GLES30.GL_KEEP, GLES30.GL_REPLACE)
        GLES30.glColorMask(false, false, false, false)
        GLES30.glDepthMask(false)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glUseProgram(progs.solid)
        GLES30.glUniform1f(progs.uSolidZ, LayerDepth.MASK)
        GLES30.glUniform4f(progs.uSolidColor, 0f, 0f, 0f, 1f)
        maskBatch.draw()

        GLES30.glColorMask(true, true, true, true)
        GLES30.glStencilFunc(GLES30.GL_EQUAL, 2, 0xFF)
        GLES30.glStencilOp(GLES30.GL_KEEP, GLES30.GL_KEEP, GLES30.GL_KEEP)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthFunc(GLES30.GL_LEQUAL)
        GLES30.glDepthMask(false)

        // ------------------------------------------------------------------------------- sclera
        scleraBatch.clear()
        if (leftActive) {
            scleraBatch.addQuad(
                leftCenterX - leftHalfWidth * 1.02f, leftCenterY + leftHalfHeight * 1.02f,
                leftCenterX + leftHalfWidth * 1.02f, leftCenterY + leftHalfHeight * 1.02f,
                leftCenterX + leftHalfWidth * 1.02f, leftCenterY - leftHalfHeight * 1.02f,
                leftCenterX - leftHalfWidth * 1.02f, leftCenterY - leftHalfHeight * 1.02f,
                0f, 1f,
            )
        }
        if (rightActive) {
            scleraBatch.addQuad(
                rightCenterX - rightHalfWidth * 1.02f, rightCenterY + rightHalfHeight * 1.02f,
                rightCenterX + rightHalfWidth * 1.02f, rightCenterY + rightHalfHeight * 1.02f,
                rightCenterX + rightHalfWidth * 1.02f, rightCenterY - rightHalfHeight * 1.02f,
                rightCenterX - rightHalfWidth * 1.02f, rightCenterY - rightHalfHeight * 1.02f,
                0f, 1f,
            )
        }
        drawOverlay(progs, scleraBatch, textures.disc, LayerDepth.EYE, 0.94f, 0.91f, 0.88f, 1f)

        // --------------------------------------------------------------------------- iris + pupil
        irisLeftBatch.clear()
        irisRightBatch.clear()
        pupilBatch.clear()
        val irisRadiusScale = 0.42f
        if (leftActive) {
            // Iris centre: use the photo's own iris landmark (468) moved by the gaze vector.
            var irisX = ndc[FaceRig.IRIS_LEFT * 2] + leftLookX * leftHalfWidth * 0.30f
            var irisY = ndc[FaceRig.IRIS_LEFT * 2 + 1] + leftLookY * leftHalfWidth * a * 0.45f
            irisX = irisX.coerceIn(
                leftCenterX - leftHalfWidth * 0.55f,
                leftCenterX + leftHalfWidth * 0.55f,
            )
            irisY = irisY.coerceIn(
                leftCenterY - leftHalfHeight * 0.45f,
                leftCenterY + leftHalfHeight * 0.45f,
            )
            val irisRadius = max(leftHalfWidth * a * irisRadiusScale, 1e-4f)
            addDisc(irisLeftBatch, irisX, irisY, irisRadius, a)
            addDisc(pupilBatch, irisX, irisY, irisRadius * 0.42f, a)
        }
        if (rightActive) {
            var irisX = ndc[FaceRig.IRIS_RIGHT * 2] + rightLookX * rightHalfWidth * 0.30f
            var irisY = ndc[FaceRig.IRIS_RIGHT * 2 + 1] + rightLookY * rightHalfWidth * a * 0.45f
            irisX = irisX.coerceIn(
                rightCenterX - rightHalfWidth * 0.55f,
                rightCenterX + rightHalfWidth * 0.55f,
            )
            irisY = irisY.coerceIn(
                rightCenterY - rightHalfHeight * 0.45f,
                rightCenterY + rightHalfHeight * 0.45f,
            )
            val irisRadius = max(rightHalfWidth * a * irisRadiusScale, 1e-4f)
            addDisc(irisRightBatch, irisX, irisY, irisRadius, a)
            addDisc(pupilBatch, irisX, irisY, irisRadius * 0.42f, a)
        }

        // Each eye uses its own crop (a turned head exposes one eye far better than the other).
        drawOverlay(progs, irisLeftBatch, textures.irisLeft, LayerDepth.EYE + 0.01f, 1f, 1f, 1f, 1f)
        drawOverlay(
            progs,
            irisRightBatch,
            if (textures.irisRight != 0) textures.irisRight else textures.irisLeft,
            LayerDepth.EYE + 0.01f,
            1f, 1f, 1f, 1f,
        )
        drawOverlay(progs, pupilBatch, textures.disc, LayerDepth.EYE + 0.02f, 0.03f, 0.03f, 0.035f, 1f)

        GLES30.glDepthMask(true)
        GLES30.glDepthFunc(GLES30.GL_LESS)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
    }

    private class Opening(
        val centerX: Float,
        val centerY: Float,
        val halfWidth: Float,
        val halfHeight: Float,
        val points: FloatArray,
    )

    /** Builds the eyelid polygon (16 points) and squashes it vertically about the eye centre. */
    private fun buildOpening(
        ndc: FloatArray,
        upper: IntArray,
        lower: IntArray,
        lidScale: Float,
    ): Opening {
        val upperCount = upper.size
        // Corner-to-corner upper ring, then the lower ring back again (endpoints shared).
        val lowerInterior = lower.size - 2
        val total = upperCount + lowerInterior
        val points = FloatArray(total * 2)

        var cx = 0f
        var cy = 0f
        for (index in upper) {
            cx += ndc[index * 2]
            cy += ndc[index * 2 + 1]
        }
        for (i in 1..lowerInterior) {
            val index = lower[i]
            cx += ndc[index * 2]
            cy += ndc[index * 2 + 1]
        }
        cx /= total
        cy /= total

        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        for (index in upper) {
            val x = ndc[index * 2]
            val y = ndc[index * 2 + 1]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        for (i in 1..lowerInterior) {
            val index = lower[i]
            val x = ndc[index * 2]
            val y = ndc[index * 2 + 1]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }

        val centerY = (minY + maxY) * 0.5f
        val restHalfHeight = ((maxY - minY) * 0.5f).coerceAtLeast(1e-4f)
        val halfWidth = ((maxX - minX) * 0.5f).coerceAtLeast(1e-4f)

        // Upper lid moves with the blink, the lower lid barely does.
        var cursor = 0
        for (index in upper) {
            points[cursor++] = ndc[index * 2]
            points[cursor++] = centerY + (ndc[index * 2 + 1] - centerY) * lidScale
        }
        for (i in 1..lowerInterior) {
            val index = lower[i]
            val lowerScale = 1f - 0.35f * (1f - lidScale)
            points[cursor++] = ndc[index * 2]
            points[cursor++] = centerY + (ndc[index * 2 + 1] - centerY) * lowerScale
        }

        return Opening(
            centerX = (minX + maxX) * 0.5f,
            centerY = centerY,
            halfWidth = halfWidth,
            halfHeight = (restHalfHeight * lidScale).coerceAtLeast(1e-4f),
            points = points,
        )
    }

    private fun addMaskFan(opening: Opening) {
        val points = opening.points
        val count = points.size / 2
        var cx = 0f
        var cy = 0f
        for (i in 0 until count) {
            cx += points[i * 2]
            cy += points[i * 2 + 1]
        }
        cx /= count
        cy /= count
        for (i in 0 until count) {
            val j = (i + 1) % count
            maskBatch.add(cx, cy)
            maskBatch.add(points[i * 2], points[i * 2 + 1])
            maskBatch.add(points[j * 2], points[j * 2 + 1])
        }
    }

    /**
     * Sprite for a circular texture (the disc texture supplies the round alpha), square in pixels:
     * [radius] is in y units, the x extent is `radius / aspect`.
     */
    private fun addDisc(batch: QuadBatch, x: Float, y: Float, radius: Float, aspect: Float) {
        val halfWidth = radius / aspect
        batch.addQuad(
            x - halfWidth, y + radius,
            x + halfWidth, y + radius,
            x + halfWidth, y - radius,
            x - halfWidth, y - radius,
            0f, 1f,
        )
    }

    private fun drawOverlay(
        progs: LayerPrograms,
        batch: QuadBatch,
        texture: Int,
        z: Float,
        tintR: Float,
        tintG: Float,
        tintB: Float,
        tintA: Float,
    ) {
        GLES30.glUseProgram(progs.overlay)
        GLES30.glUniform1f(progs.uOverlayZ, z)
        GLES30.glUniform1i(progs.uOverlayTex, 0)
        GLES30.glUniform4f(progs.uOverlayTint, tintR, tintG, tintB, tintA)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        batch.draw()
    }

    private fun shouldComposite(liveBlink: Float, photoBlink: Float, look: Float): Boolean =
        abs(photoBlink - liveBlink) > 0.05f || abs(look) > 0.05f

    /**
     * Live blink relative to the photo's own blink. A photo shot with the eyes already half closed
     * must not be closed twice, so the photo's baseline is subtracted and the remainder re-scaled.
     */
    private fun blend(coef: FloatArray, restCoef: FloatArray, index: Int): Float {
        val live = coef[index]
        val rest = restCoef[index]
        if (live <= rest) return 0f
        val headroom = (1f - rest).coerceAtLeast(0.15f)
        return ((live - rest) / headroom).coerceIn(0f, 1f)
    }

}
