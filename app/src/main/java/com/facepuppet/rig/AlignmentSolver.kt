package com.facepuppet.rig

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Closed-form 2D Umeyama similarity transform (uniform scale + rotation + translation) that maps
 * the live landmark configuration onto the photo's rest configuration.
 *
 * Why a similarity transform: it removes camera distance (scale), framing (translation) and global
 * head rotation (2D rotation) from the mimic signal. What is left over — [residuals] — is the pure
 * expression deformation, which is what drives the puppet's face warp.
 *
 * The rotation/scale are solved from a stable subset ([FaceRig.ALIGN_SUBSET]: nose bridge, inner
 * eye corners, nose tip) instead of all 478 points, so a wide smile or a yawn cannot drag the
 * registration around.
 *
 * Implementation notes:
 *  * 2D Procrustes has a closed form, so no SVD / Eigen library is needed:
 *      theta = atan2( sum(l x r), sum(l . r) )   on mean-centred pairs
 *      s     = sum( rot(l) . r ) / sum(|l|^2)
 *  * Both landmark sets are converted to an aspect-corrected isotropic space (x * w/h) before
 *    solving so that a 16:9 camera frame and a 4:3 photo can be registered against each other.
 *  * Everything is written into caller-owned arrays; no allocation on the per-frame path.
 */
class AlignmentSolver {

    /** Solved similarity transform: `R' = s * Rot(theta) * L + t`. */
    var scale: Float = 1f
        private set
    var cosTheta: Float = 1f
        private set
    var sinTheta: Float = 0f
        private set
    var translateX: Float = 0f
        private set
    var translateY: Float = 0f
        private set

    /** False when the solve was degenerate (no face, collinear points, ...). */
    var valid: Boolean = false
        private set

    /** Solves `live -> rest`. Returns true when a usable transform was produced. */
    fun solve(
        live: FloatArray,
        rest: FloatArray,
        indices: IntArray,
        liveAspect: Float,
        restAspect: Float,
    ): Boolean {
        val n = indices.size
        if (n < 3) {
            valid = false
            return false
        }
        var sumLx = 0f
        var sumLy = 0f
        var sumRx = 0f
        var sumRy = 0f
        for (index in indices) {
            sumLx += live[index * 3] * liveAspect
            sumLy += live[index * 3 + 1]
            sumRx += rest[index * 3] * restAspect
            sumRy += rest[index * 3 + 1]
        }
        val invN = 1f / n
        val meanLx = sumLx * invN
        val meanLy = sumLy * invN
        val meanRx = sumRx * invN
        val meanRy = sumRy * invN

        var cross = 0f   // sum( l x r )
        var dot = 0f     // sum( l . r )
        var normL = 0f   // sum( |l|^2 )
        for (index in indices) {
            val lx = live[index * 3] * liveAspect - meanLx
            val ly = live[index * 3 + 1] - meanLy
            val rx = rest[index * 3] * restAspect - meanRx
            val ry = rest[index * 3 + 1] - meanRy
            cross += lx * ry - ly * rx
            dot += lx * rx + ly * ry
            normL += lx * lx + ly * ly
        }
        if (normL < 1e-9f || dot == 0f && cross == 0f) {
            valid = false
            return false
        }
        val theta = atan2(cross, dot)
        val c = cos(theta)
        val s = sin(theta)

        var rotatedDot = 0f
        for (index in indices) {
            val lx = live[index * 3] * liveAspect - meanLx
            val ly = live[index * 3 + 1] - meanLy
            val rx = rest[index * 3] * restAspect - meanRx
            val ry = rest[index * 3 + 1] - meanRy
            rotatedDot += (c * lx - s * ly) * rx + (s * lx + c * ly) * ry
        }
        val solvedScale = rotatedDot / normL
        if (!solvedScale.isFinite() || solvedScale <= 1e-6f) {
            valid = false
            return false
        }

        scale = solvedScale
        cosTheta = c
        sinTheta = s
        translateX = meanRx - solvedScale * (c * meanLx - s * meanLy)
        translateY = meanRy - solvedScale * (s * meanLx + c * meanLy)
        valid = true
        return true
    }

    /**
     * Maps live landmark [index] into rest (photo) space, writing x/y into [out] at `2 * slot`.
     * Used by [residuals]; kept public so the renderer can project anchors as well.
     */
    fun mapX(x: Float, y: Float, aspect: Float): Float =
        scale * (cosTheta * (x * aspect) - sinTheta * y) + translateX

    fun mapY(x: Float, y: Float, aspect: Float): Float =
        scale * (sinTheta * (x * aspect) + cosTheta * y) + translateY

    /**
     * Writes the clamped expression residual `D_i = T(L_i) - R_i` for every landmark.
     *
     * @param out `2 * FaceFrame.LANDMARK_COUNT` floats: dx, dy per landmark, in **face units**
     *        (photo IOD == 1) after [toFaceUnits].
     * @param clamp max residual length in face units (spec: 6% of the inter-ocular distance).
     * @param toFaceUnits `1 / photoInterOcular` — converts photo-normalised units to face units.
     * @param weight damping applied to the residual, so the residual and the blendshape rig share
     *        the deformation budget instead of both applying the full jaw drop.
     */
    fun residuals(
        live: FloatArray,
        rest: FloatArray,
        liveAspect: Float,
        restAspect: Float,
        out: FloatArray,
        clamp: Float,
        toFaceUnits: Float,
        weight: Float,
        landmarkCount: Int,
    ) {
        if (!valid) {
            java.util.Arrays.fill(out, 0f)
            return
        }
        val limit = clamp / if (weight > 1e-5f) weight else 1f
        for (i in 0 until landmarkCount) {
            val lx = live[i * 3]
            val ly = live[i * 3 + 1]
            val mappedX = mapX(lx, ly, liveAspect)
            val mappedY = mapY(lx, ly, liveAspect)
            var dx = (mappedX - rest[i * 3] * restAspect) * toFaceUnits
            var dy = (mappedY - rest[i * 3 + 1]) * toFaceUnits
            val length = sqrt(dx * dx + dy * dy)
            if (length > limit) {
                val k = limit / length
                dx *= k
                dy *= k
            }
            out[i * 2] = dx * weight
            out[i * 2 + 1] = dy * weight
        }
    }
}
