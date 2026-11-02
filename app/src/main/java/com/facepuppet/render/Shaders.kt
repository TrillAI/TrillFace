package com.facepuppet.render

/**
 * Every GLSL ES 3.00 program used by the puppet, as source strings.
 *
 * Program inventory (uniform names are referenced from the renderer):
 *  * [PUPPET]        2.5D head mesh + extruded silhouette skirt.
 *  * [SOLID]         flat-coloured NDC triangles (sclera, pupils, stencil fills).
 *  * [GRADIENT]      NDC triangles with a per-vertex blend factor (mouth interior, background).
 *  * [OVERLAY]       textured NDC triangles (teeth, tongue, iris).
 *
 * All layer geometry is expressed directly in normalised device coordinates, so the overlay
 * programs take a constant `u_z` instead of a matrix: the mouth/eye layers are built on the CPU from
 * the already-projected warped landmarks, which keeps the stencil mask and the visible geometry
 * pixel-identical by construction.
 */
object Shaders {

    // ---------------------------------------------------------------------------------------------
    // 1. Puppet mesh
    // ---------------------------------------------------------------------------------------------

    const val PUPPET_VERTEX = """#version 300 es
layout(location = 0) in vec3 a_rest;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec3 a_delta;
layout(location = 3) in float a_shade;

uniform mat4 u_mvp;

out vec2 v_uv;
out float v_shade;

void main() {
    // Expression warp first, head pose after: the UVs are fixed per vertex, so the painted skin
    // travels with the geometry while R_pose supplies the turn and the parallax.
    vec3 p = a_rest + a_delta;
    v_uv = a_uv;
    v_shade = a_shade;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

    const val PUPPET_FRAGMENT = """#version 300 es
precision mediump float;

in vec2 v_uv;
in float v_shade;

uniform sampler2D u_tex;
uniform vec3 u_rim;

out vec4 fragColor;

void main() {
    vec4 skin = texture(u_tex, v_uv);
    float skirt = clamp(v_shade, 0.0, 1.0);
    // The skirt is the photo's own rim colour darkened 45% (done on the CPU); 0.92 keeps a hint of
    // the sampled boundary colour so the cut-away edge does not read as flat plastic.
    vec3 shaded = mix(skin.rgb, u_rim, skirt * 0.92);
    fragColor = vec4(shaded, 1.0);
}
"""

    // ---------------------------------------------------------------------------------------------
    // 2. Flat colour (stencil fills, sclera, pupils)
    // ---------------------------------------------------------------------------------------------

    const val SOLID_VERTEX = """#version 300 es
layout(location = 0) in vec2 a_pos;

uniform float u_z;

void main() {
    gl_Position = vec4(a_pos, u_z, 1.0);
}
"""

    const val SOLID_FRAGMENT = """#version 300 es
precision mediump float;

uniform vec4 u_color;

out vec4 fragColor;

void main() {
    fragColor = u_color;
}
"""

    // ---------------------------------------------------------------------------------------------
    // 3. Vertical gradient (mouth interior, background)
    // ---------------------------------------------------------------------------------------------

    const val GRADIENT_VERTEX = """#version 300 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_t;

uniform float u_z;

out float v_t;

void main() {
    v_t = a_t;
    gl_Position = vec4(a_pos, u_z, 1.0);
}
"""

    const val GRADIENT_FRAGMENT = """#version 300 es
precision mediump float;

in float v_t;

uniform vec4 u_top;
uniform vec4 u_bottom;

out vec4 fragColor;

void main() {
    fragColor = mix(u_top, u_bottom, clamp(v_t, 0.0, 1.0));
}
"""

    // ---------------------------------------------------------------------------------------------
    // 4. Textured overlay (teeth, tongue, iris)
    // ---------------------------------------------------------------------------------------------

    const val OVERLAY_VERTEX = """#version 300 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;

uniform float u_z;

out vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, u_z, 1.0);
}
"""

    const val OVERLAY_FRAGMENT = """#version 300 es
precision mediump float;

in vec2 v_uv;

uniform sampler2D u_tex;
uniform vec4 u_tint;

out vec4 fragColor;

void main() {
    vec4 texel = texture(u_tex, v_uv) * u_tint;
    if (texel.a < 0.004) discard;
    fragColor = texel;
}
"""
}
