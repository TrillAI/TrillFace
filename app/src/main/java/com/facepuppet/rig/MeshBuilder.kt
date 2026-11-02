package com.facepuppet.rig

import java.io.InputStream
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The canonical 468-vertex MediaPipe face mesh, normalised once to *face units*:
 * origin at the face centre, +x right, +y up, +z towards the viewer, inter-ocular distance == 1.
 */
class CanonicalMesh(
    /** 468 x 3 positions in face units. */
    val positions: FloatArray,
    /** 898 x 3 triangle indices (from the OBJ `f` list, 0-based). */
    val faces: IntArray,
) {
    val vertexCount: Int get() = positions.size / 3
}

/**
 * Render-ready geometry for one photo: the canonical topology carrying the photo's rest positions,
 * fixed UVs, per-vertex skirt shading weight, plus the extruded silhouette skirt.
 *
 * The vertex buffer is interleaved as `[x, y, z, u, v, shade]` (6 floats / 24 bytes per vertex):
 * `a_rest` (3), `a_uv` (2), `a_shade` (1). Per-frame deformation arrives in a second, dynamic
 * `[dx, dy, dz]` buffer that is added inside the vertex shader.
 */
class MeshData(
    val vertexCount: Int,
    val rest: FloatArray,
    val uv: FloatArray,
    val shade: FloatArray,
    val indices: IntArray,
    /** Number of vertices that correspond to real landmarks (delta buffer is sized with this). */
    val landmarkVertexCount: Int,
    /** Skirt vertex i takes its deformation from `skirtSource[i]`. */
    val skirtSource: IntArray,
    val skirtTarget: IntArray,
    /** Distance between the eye corners in the photo's face units (always 1.0 by construction). */
    val interOcular: Float,
    /** Width of the photo face oval in face units, used to size teeth / depth. */
    val faceWidth: Float,
) {
    /** Floats per interleaved vertex. */
    val strideFloats: Int get() = 6
}

object MeshBuilder {

    private const val CANONICAL_VERTICES = 468
    private const val CANONICAL_TRIANGLES = 898

    /** Fraction of the face width used as total head depth (spec: 0.35 x face width). */
    private const val DEPTH_RATIO = 0.35f

    /** How far the silhouette skirt is pushed backwards along -z, in face units. */
    private const val SKIRT_DEPTH = 0.62f

    /** The back ring is pulled slightly inwards so the cut-away looks like a head, not a slab. */
    private const val SKIRT_TAPER = 0.95f

    // ---------------------------------------------------------------------------------------------
    // OBJ parsing
    // ---------------------------------------------------------------------------------------------

    /**
     * Parses `canonical_face_model.obj` (468 `v` lines and the 898-triangle `f` list) and
     * normalises it to face units.
     *
     * The OBJ ships with its own triangle list — we use it verbatim rather than re-triangulating,
     * which keeps the flat topology identical to the one MediaPipe's own face geometry module uses.
     */
    fun loadCanonical(stream: InputStream): CanonicalMesh {
        val raw = FloatArray(CANONICAL_VERTICES * 3)
        var vertexCount = 0
        val faces = IntArray(CANONICAL_TRIANGLES * 3)
        var faceIndexCount = 0

        stream.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (line.isEmpty()) continue
                when {
                    line.startsWith("v ") -> {
                        val parts = line.trim().split(' ', '\t')
                        if (parts.size < 4) continue
                        if (vertexCount >= CANONICAL_VERTICES) {
                            throw IllegalStateException(
                                "canonical_face_model.obj has more than $CANONICAL_VERTICES vertices",
                            )
                        }
                        raw[vertexCount * 3] = parts[1].toFloat()
                        raw[vertexCount * 3 + 1] = parts[2].toFloat()
                        raw[vertexCount * 3 + 2] = parts[3].toFloat()
                        vertexCount++
                    }

                    line.startsWith("f ") -> {
                        val parts = line.trim().split(' ', '\t')
                        if (parts.size < 4) continue
                        if (faceIndexCount + 3 > faces.size) {
                            throw IllegalStateException(
                                "canonical_face_model.obj has more than $CANONICAL_TRIANGLES triangles",
                            )
                        }
                        for (k in 1..3) {
                            // Tokens look like "174/43" or "174/43/12": the vertex index is first.
                            val token = parts[k]
                            val slash = token.indexOf('/')
                            val number = if (slash >= 0) token.substring(0, slash) else token
                            val oneBased = number.toInt()
                            faces[faceIndexCount++] = oneBased - 1
                        }
                    }
                }
            }
        }

        if (vertexCount != CANONICAL_VERTICES || faceIndexCount != CANONICAL_TRIANGLES * 3) {
            throw IllegalStateException(
                "canonical_face_model.obj is not the expected mesh: " +
                    "$vertexCount vertices (want $CANONICAL_VERTICES), " +
                    "${faceIndexCount / 3} triangles (want $CANONICAL_TRIANGLES)",
            )
        }
        for (index in faces) {
            if (index < 0 || index >= CANONICAL_VERTICES) {
                throw IllegalStateException("canonical_face_model.obj has an out-of-range face index $index")
            }
        }
        return CanonicalMesh(normalise(raw), faces)
    }

    /** Normalises raw OBJ coordinates to face units (pivot inside the head so yaw has parallax). */
    private fun normalise(raw: FloatArray): FloatArray {
        val out = FloatArray(raw.size)

        // Scale: eye corner distance -> 1.0
        val leftEye = FaceRig.EYE_CORNER_LEFT
        val rightEye = FaceRig.EYE_CORNER_RIGHT
        val iodRaw = distance(raw, leftEye, rightEye).coerceAtLeast(1e-5f)
        val scale = 1f / iodRaw

        // Pivot: between the eyes in x, at the eye line in y, mean oval depth in z.
        val cx = (raw[leftEye * 3] + raw[rightEye * 3]) * 0.5f
        val cy = (raw[leftEye * 3 + 1] + raw[rightEye * 3 + 1]) * 0.5f

        var zSum = 0f
        for (index in FaceRig.FACE_OVAL) zSum += raw[index * 3 + 2]
        val cz = zSum / FaceRig.FACE_OVAL.size

        // Vertical squash of the depth axis: total head depth becomes DEPTH_RATIO * face width.
        var minZ = Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE
        for (v in 0 until CANONICAL_VERTICES) {
            val z = raw[v * 3 + 2]
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
        }
        val zSpan = ((maxZ - minZ) * scale).coerceAtLeast(1e-4f)
        val faceWidth = (abs(raw[454 * 3] - raw[234 * 3])) * scale
        val zScale = (DEPTH_RATIO * faceWidth) / zSpan

        for (v in 0 until CANONICAL_VERTICES) {
            out[v * 3] = (raw[v * 3] - cx) * scale
            // MediaPipe's canonical mesh is y-up already; keep it.
            out[v * 3 + 1] = (raw[v * 3 + 1] - cy) * scale
            out[v * 3 + 2] = (raw[v * 3 + 2] - cz) * scale * zScale
        }
        return out
    }

    private fun distance(positions: FloatArray, a: Int, b: Int): Float {
        val dx = positions[a * 3] - positions[b * 3]
        val dy = positions[a * 3 + 1] - positions[b * 3 + 1]
        val dz = positions[a * 3 + 2] - positions[b * 3 + 2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    // ---------------------------------------------------------------------------------------------
    // Photo mesh
    // ---------------------------------------------------------------------------------------------

    /**
     * Builds the render geometry for [photo]: x/y come from the photo's rest landmarks (so the
     * puppet has the photo's face shape), z comes from the canonical mesh (so it has real depth),
     * and UVs are the photo landmark positions themselves.
     */
    fun build(canonical: CanonicalMesh, photo: PhotoRig): MeshData {
        val landmarks = CANONICAL_VERTICES
        val skirtVerts = FaceRig.FACE_OVAL.size
        val vertexCount = landmarks + skirtVerts

        val rest = FloatArray(vertexCount * 3)
        val uv = FloatArray(vertexCount * 2)
        val shade = FloatArray(vertexCount)

        val aspect = photo.aspect
        val invIod = 1f / photo.interOcular.coerceAtLeast(1e-4f)
        val cx = photo.centerX * aspect
        val cy = photo.centerY

        for (v in 0 until landmarks) {
            val px = photo.restLandmarks[v * 3]
            val py = photo.restLandmarks[v * 3 + 1]
            rest[v * 3] = (px * aspect - cx) * invIod
            rest[v * 3 + 1] = -(py - cy) * invIod
            rest[v * 3 + 2] = canonical.positions[v * 3 + 2]
            uv[v * 2] = px
            uv[v * 2 + 1] = py
            shade[v] = 0f
        }

        // Silhouette skirt: duplicate the face-oval ring, push it back along -z and taper it in.
        val skirtSource = IntArray(skirtVerts)
        val skirtTarget = IntArray(skirtVerts)
        for (k in 0 until skirtVerts) {
            val src = FaceRig.FACE_OVAL[k]
            val dst = landmarks + k
            skirtSource[k] = src
            skirtTarget[k] = dst
            rest[dst * 3] = rest[src * 3] * SKIRT_TAPER
            rest[dst * 3 + 1] = rest[src * 3 + 1] * SKIRT_TAPER
            rest[dst * 3 + 2] = rest[src * 3 + 2] - SKIRT_DEPTH
            uv[dst * 2] = uv[src * 2]
            uv[dst * 2 + 1] = uv[src * 2 + 1]
            shade[dst] = 1f
        }

        // Topology: canonical triangles + two triangles per skirt segment.
        val indices = IntArray(canonical.faces.size + skirtVerts * 6)
        canonical.faces.copyInto(indices, 0)
        var cursor = canonical.faces.size
        for (k in 0 until skirtVerts) {
            val next = (k + 1) % skirtVerts
            val frontA = FaceRig.FACE_OVAL[k]
            val frontB = FaceRig.FACE_OVAL[next]
            val backA = skirtTarget[k]
            val backB = skirtTarget[next]
            indices[cursor++] = frontA
            indices[cursor++] = backA
            indices[cursor++] = backB
            indices[cursor++] = frontA
            indices[cursor++] = backB
            indices[cursor++] = frontB
        }

        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        for (index in FaceRig.FACE_OVAL) {
            val x = rest[index * 3]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
        }

        return MeshData(
            vertexCount = vertexCount,
            rest = rest,
            uv = uv,
            shade = shade,
            indices = indices,
            landmarkVertexCount = landmarks,
            skirtSource = skirtSource,
            skirtTarget = skirtTarget,
            interOcular = 1f,
            faceWidth = (maxX - minX).coerceAtLeast(0.5f),
        )
    }
}
