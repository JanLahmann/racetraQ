"""Write racetraq.org's bitmap icons from the mark in browser/index.html.

Run as ``python tools/make_icons.py`` (Pillow). The mark is the inline SVG
favicon: a dark rounded square, a purple triangle, a white dot, on a 32-unit
grid. This draws the same shapes 8x oversampled and scales down, writing

- ``browser/public/apple-touch-icon.png``: 180x180, opaque, square corners
  (iOS rounds them itself);
- ``browser/public/favicon.ico``: 16, 32 and 48 px, for browsers and tools
  that ask for /favicon.ico instead of reading the SVG link.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PUBLIC = Path(__file__).resolve().parent.parent / "browser" / "public"

BG = (15, 17, 23)  # #0f1117
QUANTUM = (122, 92, 255)  # #7a5cff
WHITE = (255, 255, 255)
OVERSAMPLE = 8


def mark(size: int, rounded: bool) -> Image.Image:
    """The mark at ``size`` px; ``rounded`` keeps the SVG's rx=7 corners
    (transparent outside), otherwise the background fills the square."""
    big = size * OVERSAMPLE
    u = big / 32  # one SVG unit
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0) if rounded else BG + (255,))
    draw = ImageDraw.Draw(img)
    if rounded:
        draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=7 * u, fill=BG)
    draw.polygon([(7 * u, 22 * u), (16 * u, 7 * u), (25 * u, 22 * u)], fill=QUANTUM)
    r = 2.6 * u
    draw.ellipse((16 * u - r, 17 * u - r, 16 * u + r, 17 * u + r), fill=WHITE)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    mark(180, rounded=False).convert("RGB").save(PUBLIC / "apple-touch-icon.png", optimize=True)
    sizes = [16, 32, 48]
    mark(48, rounded=True).save(
        PUBLIC / "favicon.ico", sizes=[(s, s) for s in sizes],
        append_images=[mark(s, rounded=True) for s in sizes[:-1]],
    )
    for name in ("apple-touch-icon.png", "favicon.ico"):
        print(PUBLIC / name)


if __name__ == "__main__":
    main()
