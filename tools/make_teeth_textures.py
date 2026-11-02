#!/usr/bin/env python3
"""Generates the two 1024x256 RGBA teeth atlases used by the mouth layer.

Run from the repository root:

    python3 tools/make_teeth_textures.py

Outputs:
    app/src/main/res/drawable-nodpi/teeth_upper.png   (gum line at v = 0, tooth tips at v = 1)
    app/src/main/res/drawable-nodpi/teeth_lower.png   (tooth tips at v = 0, gum line at v = 1)

Both strips are drawn as separate rows by MouthLayer, so the atlas is a single
horizontal band of teeth: individual teeth are separated by soft, slightly
darker inter-proximal gaps and a specular highlight band sits just under the
gum line, which is what makes a flat strip read as teeth when it is stretched
over ~50% of the inter-ocular distance.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter

SS = 2  # supersampling factor
W, H = 1024, 256
OUT_DIR = os.path.join("app", "src", "main", "res", "drawable-nodpi")

# Tooth layout: relative widths, a small height jitter and a tiny tilt give the
# row an organic rhythm instead of a picket fence.
UPPER_WIDTHS = [1.00, 0.82, 1.26, 1.22, 1.00, 1.00, 1.22, 1.26, 0.82, 1.00]
UPPER_HEIGHTS = [0.86, 0.94, 1.00, 1.00, 0.93, 0.93, 1.00, 1.00, 0.94, 0.86]
LOWER_WIDTHS = [0.90,              # lower central incisors
                0.78, 0.72,        # lateral incisors
                0.95, 0.95,        # canines
                1.06, 1.02,        # first premolars
                0.98, 0.96,        # second premolars
                0.92, 0.90,        # first molars
                0.78, 0.72]        # second molars (partially out of frame)
LOWER_HEIGHTS = [0.96, 0.98, 1.00, 1.00, 0.97, 0.95, 0.94, 0.92, 0.91, 0.88, 0.84, 0.80]

ENAMEL_TOP = (250, 248, 241)
ENAMEL_MID = (238, 233, 221)
ENAMEL_LOW = (214, 206, 191)
GUM = (196, 122, 122)


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _shade(base, factor):
    return tuple(max(0, min(255, int(round(c * factor)))) for c in base)


def _tooth_polygon(cx, top, bottom, half_w, tilt, bulge):
    """A rounded, slightly tapered tooth silhouette (8 points, smooth-ish)."""
    h = bottom - top
    pts = []
    # left side, top -> bottom
    pts.append((cx - half_w * 0.92 + tilt * 0.15, top))
    pts.append((cx - half_w + tilt * 0.55, top + h * 0.34))
    pts.append((cx - half_w * 0.97 + tilt * 0.85, top + h * 0.72))
    pts.append((cx - half_w * bulge + tilt, bottom))
    # rounded occlusal edge
    pts.append((cx - half_w * 0.34 + tilt, bottom + h * 0.045))
    pts.append((cx + half_w * 0.34 + tilt, bottom + h * 0.045))
    # right side, bottom -> top
    pts.append((cx + half_w * bulge + tilt, bottom))
    pts.append((cx + half_w * 0.97 + tilt * 0.85, top + h * 0.72))
    pts.append((cx + half_w + tilt * 0.55, top + h * 0.34))
    pts.append((cx + half_w * 0.92 + tilt * 0.15, top))
    return pts


def _build_row(widths, heights, gum_at_top: bool) -> Image.Image:
    w, h = W * SS, H * SS
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    gum_h = int(h * 0.13)
    total = sum(widths)
    # The strip is intentionally a little wider than the frame so the row can be
    # mirrored/extended without a visible seam at the mouth corners.
    span = w * 1.02
    x = -w * 0.01

    for i, (rel_w, rel_h) in enumerate(zip(widths, heights)):
        tw = span * (rel_w / total)
        if gum_at_top:
            top = int(gum_h * 0.55)
            bottom = int(h * (0.34 + 0.44 * rel_h))
        else:
            top = int(h * (1.0 - (0.34 + 0.44 * rel_h)))
            bottom = h - int(gum_h * 0.55)

        cx = x + tw * 0.5
        tilt = (-1.0 if i < len(widths) / 2 else 1.0) * tw * 0.012 * (1.0 - abs(i - len(widths) / 2) / len(widths))
        half_w = tw * 0.5 * 0.90
        bulge = 0.90 + 0.05 * math.sin(i * 1.7)

        poly = _tooth_polygon(cx, top, bottom, half_w, tilt, bulge)

        # Vertical enamel shading: bright at the incisal edge, warmer near the gum.
        tooth = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(tooth)
        d.polygon(poly, fill=ENAMEL_MID + (255,))
        top_c = ENAMEL_TOP if gum_at_top else ENAMEL_LOW
        bot_c = ENAMEL_LOW if gum_at_top else ENAMEL_TOP
        grad = ImageDraw.Draw(tooth)
        steps = 48
        for s in range(steps):
            t0 = s / steps
            t1 = (s + 1) / steps
            y0 = top + (bottom - top) * t0
            y1 = top + (bottom - top) * t1
            grad.rectangle(
                [cx - tw, y0, cx + tw, y1 + 1],
                fill=_lerp(top_c, bot_c, t0) + (255,),
            )
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        tooth.putalpha(mask)

        # Soft inter-proximal shadow on each side.
        edge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ed = ImageDraw.Draw(edge)
        ed.polygon(poly, fill=_shade(ENAMEL_MID, 0.90) + (255,))
        edge = edge.filter(ImageFilter.GaussianBlur(SS * 2.2))
        edge.putalpha(edge.getchannel("A").point(lambda v: int(v * 0.55)))
        tooth = Image.alpha_composite(tooth, Image.composite(edge, Image.new("RGBA", (w, h), (0, 0, 0, 0)), mask))

        img = Image.alpha_composite(img, tooth)
        x += tw

    # Dark backing behind the tooth row: inter-proximal gaps must read as
    # shadow inside the mouth, never as holes that let the interior show through.
    backing = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(backing)
    if gum_at_top:
        bd.rectangle([0, int(gum_h * 0.4), w, int(h * 0.80)], fill=(34, 20, 20, 232))
    else:
        bd.rectangle([0, int(h * 0.18), w, h - int(gum_h * 0.4)], fill=(34, 20, 20, 232))
    backing = backing.filter(ImageFilter.GaussianBlur(SS * 1.2))
    img = Image.alpha_composite(backing, img)

    # Gum line: a soft pink band on the gum side, feathered so it fades out.
    gum = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gum)
    if gum_at_top:
        gd.rectangle([0, 0, w, gum_h], fill=GUM + (255,))
    else:
        gd.rectangle([0, h - gum_h, w, h], fill=GUM + (255,))
    gum = gum.filter(ImageFilter.GaussianBlur(SS * 3.0))
    gum.putalpha(gum.getchannel("A").point(lambda v: int(v * 0.92)))
    img = Image.alpha_composite(img, gum)

    # Specular highlight row: a bright horizontal band just under the gum line.
    hi = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi)
    if gum_at_top:
        band = [0, int(h * 0.16), w, int(h * 0.20)]
    else:
        band = [0, int(h * 0.80), w, int(h * 0.84)]
    hd.rectangle(band, fill=(255, 255, 255, 96))
    hi = hi.filter(ImageFilter.GaussianBlur(SS * 2.4))
    # Confine the highlight to the teeth silhouette.
    hi.putalpha(Image.composite(hi.getchannel("A"), Image.new("L", (w, h), 0), img.getchannel("A").point(lambda v: 255 if v > 24 else 0)))
    img = Image.alpha_composite(img, hi)

    return img.resize((W, H), Image.LANCZOS)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    upper = _build_row(UPPER_WIDTHS, UPPER_HEIGHTS, gum_at_top=True)
    lower = _build_row(LOWER_WIDTHS, LOWER_HEIGHTS, gum_at_top=False)
    for name, im in (("teeth_upper.png", upper), ("teeth_lower.png", lower)):
        path = os.path.join(OUT_DIR, name)
        im.save(path, optimize=True)
        print("wrote", path, im.size, im.mode)


if __name__ == "__main__":
    main()
