package com.facepuppet.render

import android.opengl.GLES30
import com.facepuppet.rig.FaceRig
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The mouth: stencil mask, dark interior, tongue and the two **separate** teeth rows.
 *
 * Pass order (spec 5):
 *  1. write the inner-lip polygon into the stencil buffer,
 *  2. dark maroon interior gradient, drawn only where the stencil passes,
 *  3. tongue sprite anchored to the floor of the mouth (drawn first so both teeth rows occlude it),
 *  4. upper teeth strip anchored to the *upper* inner-lip edge — rides with the skull,
 *  5. lower teeth strip anchored to the *lower* inner-lip edge — follows the jaw.
 *
 * Everything is built in NDC on the CPU from the already-projected, already-deformed landmark
 * positions. That guarantees the mask and the content are the same shape, and it means the teeth
 * can never detach from the lips: both anchors are literally the lip landmarks.
 *
 * Allocation: the batches below are pre-allocated and reused; `draw` allocates nothing.
 */
class MouthLayer {

    private val maskBatch = TriBatch(48)
    private val interiorBatch = QuadBatch(6)
    private val tongueBatch = QuadBatch(6)
    private val upperBatch = QuadBatch(6)
    private val lowerBatch = QuadBatch(6)

    fun onSurfaceCreated() {
        maskBatch.create()
        interiorBatch.create()
        tongueBatch.create()
        upperBatch.create()
        lowerBatch.create()
    }

    fun release() {
        maskBatch.destroy()
        interiorBatch.destroy()
        tongueBatch.destroy()
        upperBatch.destroy()
        lowerBatch.destroy()
    }

    /**
     * @param ndc 478 projected positions (x, y) in NDC of the *deformed and posed* puppet.
     * @param coef 52 blendshape coefficients.
     * @param aspect viewport width / height. NDC x and y are *not* isotropic, so every synthetic
     *        quad is laid out in "y units" (multiply an NDC x length by [aspect]) and converted
     *        back when it is written out. Without this, teeth would be stretched vertically on a
     *        20:9 phone and squashed on a tablet.
     */
    fun draw(
        progs: LayerPrograms,
        ndc: FloatArray,
        coef: FloatArray,
        textures: LayerTextures,
        aspect: Float,
    ) {
        val a = aspect.coerceIn(0.2f, 5f)
        val inner = FaceRig.LIPS_INNER
        var cx = 0f
        var cy = 0f
        for (index in inner) {
            cx += ndc[index * 2]
            cy += ndc[index * 2 + 1]
        }
        val invCount = 1f / inner.size
        cx *= invCount
        cy *= invCount

        val iod = sqrt(
            sq((ndc[FaceRig.EYE_CORNER_LEFT * 2] - ndc[FaceRig.EYE_CORNER_RIGHT * 2]) * a) +
                sq(ndc[FaceRig.EYE_CORNER_LEFT * 2 + 1] - ndc[FaceRig.EYE_CORNER_RIGHT * 2 + 1]),
        ).coerceAtLeast(1e-4f)

        val upperLipY = ndc[FaceRig.LIP_CENTER_TOP * 2 + 1]
        val lowerLipY = ndc[FaceRig.LIP_CENTER_BOTTOM * 2 + 1]
        val gap = (upperLipY - lowerLipY).coerceAtLeast(0f)
        val mouthWidth = abs(ndc[FaceRig.LIP_INNER_RIGHT * 2] - ndc[FaceRig.LIP_INNER_LEFT * 2]) * a
        val mouthCenterX = (ndc[FaceRig.LIP_INNER_RIGHT * 2] + ndc[FaceRig.LIP_INNER_LEFT * 2]) * 0.5f
        val jawOpen = coef[FaceRig.B_JAW_OPEN]

        // ---------------------------------------------------------------------------- stencil mask
        maskBatch.clear()
        val n = inner.size
        for (k in 0 until n) {
            val a = inner[k]
            val b = inner[(k + 1) % n]
            maskBatch.add(cx, cy)
            maskBatch.add(ndc[a * 2], ndc[a * 2 + 1])
            maskBatch.add(ndc[b * 2], ndc[b * 2 + 1])
        }

        GLES30.glEnable(GLES30.GL_STENCIL_TEST)
        GLES30.glStencilMask(0xFF)
        GLES30.glStencilFunc(GLES30.GL_ALWAYS, 1, 0xFF)
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
        GLES30.glStencilFunc(GLES30.GL_EQUAL, 1, 0xFF)
        GLES30.glStencilOp(GLES30.GL_KEEP, GLES30.GL_KEEP, GLES30.GL_KEEP)
        // The teeth get a negative Z bias (towards the camera) and LEQUAL so they can never z-fight
        // with the lips they are anchored to; depth writes stay off so the layers stack purely by
        // draw order.
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthFunc(GLES30.GL_LEQUAL)
        GLES30.glDepthMask(false)

        // ------------------------------------------------------------------------ mouth interior
        val halfWidth = mouthWidth * 0.62f
        val halfHeight = (gap * 0.75f).coerceAtLeast(iod * 0.012f)
        interiorBatch.clear()
        interiorBatch.addQuad(
            cx - halfWidth / a, cy + halfHeight,
            cx + halfWidth / a, cy + halfHeight,
            cx + halfWidth / a, cy - halfHeight,
            cx - halfWidth / a, cy - halfHeight,
            /* t at the top */ 0f, 1f,
        )
        GLES30.glUseProgram(progs.gradient)
        GLES30.glUniform1f(progs.uGradZ, LayerDepth.INTERIOR)
        GLES30.glUniform4f(progs.uGradTop, 0.10f, 0.020f, 0.028f, 1f)
        GLES30.glUniform4f(progs.uGradBottom, 0.26f, 0.065f, 0.080f, 1f)
        interiorBatch.drawScalarT()

        // ------------------------------------------------------------------------------- tongue
        val tongueHeight = (gap * 0.62f).coerceAtLeast(iod * 0.006f)
        val tongueWidth = (mouthWidth * 0.52f).coerceAtLeast(iod * 0.05f)
        val tongueCenterY = lowerLipY + tongueHeight * 0.30f
        tongueBatch.clear()
        tongueBatch.addQuad(
            cx - tongueWidth * 0.5f / a, tongueCenterY + tongueHeight * 0.5f,
            cx + tongueWidth * 0.5f / a, tongueCenterY + tongueHeight * 0.5f,
            cx + tongueWidth * 0.5f / a, tongueCenterY - tongueHeight * 0.5f,
            cx - tongueWidth * 0.5f / a, tongueCenterY - tongueHeight * 0.5f,
            0f, 1f,
        )
        drawOverlay(progs, tongueBatch, textures.tongue, LayerDepth.TONGUE, 1f, 0.86f, 0.86f, 1f)

        // ---------------------------------------------------------------------------- teeth rows
        // Strip width: 0.55 x IOD, never wider than the mouth itself.
        val stripWidth = minOf(iod * 0.55f, mouthWidth * 1.10f).coerceAtLeast(iod * 0.10f)
        val fullHeight = stripWidth * 0.30f
        // Each row shows at most 55% of the opening, so the rows never meet in the middle and the
        // dark interior + tongue stay visible between them.
        val visible = minOf(fullHeight, gap * 0.55f)
        val frac = (visible / fullHeight).coerceIn(0f, 1f)

        if (gap > iod * 0.006f && frac > 0.01f) {
            // Upper row: gum line on the upper inner-lip edge, tooth tips hanging down.
            upperBatch.clear()
            upperBatch.addQuad(
                mouthCenterX - stripWidth * 0.5f / a, upperLipY,
                mouthCenterX + stripWidth * 0.5f / a, upperLipY,
                mouthCenterX + stripWidth * 0.5f / a, upperLipY - visible,
                mouthCenterX - stripWidth * 0.5f / a, upperLipY - visible,
                0f, frac,
            )
            drawOverlay(progs, upperBatch, textures.teethUpper, LayerDepth.TEETH, 1f, 1f, 1f, 1f)

            // Lower row: tips on the lower inner-lip edge, gum line below it. The extra 1.6 x jaw
            // offset keeps the row glued to the lip as the mouth opens wide.
            val jawOffset = jawOpen * 1.6f * iod * 0.015f
            val lowerTop = lowerLipY - jawOffset
            lowerBatch.clear()
            lowerBatch.addQuad(
                mouthCenterX - stripWidth * 0.5f / a, lowerTop + visible,
                mouthCenterX + stripWidth * 0.5f / a, lowerTop + visible,
                mouthCenterX + stripWidth * 0.5f / a, lowerTop,
                mouthCenterX - stripWidth * 0.5f / a, lowerTop,
                0f, frac,
            )
            drawOverlay(progs, lowerBatch, textures.teethLower, LayerDepth.TEETH, 1f, 1f, 1f, 1f)
        }

        GLES30.glDepthMask(true)
        GLES30.glDepthFunc(GLES30.GL_LESS)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
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

    private fun sq(v: Float) = v * v
}
