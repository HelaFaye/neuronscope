"""
Pixel art and cartoons for CLIP-style models: preprocessing and prompts.

CLIP is trained mostly on photos. Two things hurt it on sprites and cartoons
before the model sees anything:

- the processor resizes with bicubic filtering, which smears a 32x32 sprite
  into a blurry blob, and then centre-crops, which cuts off whatever is not in
  the middle of a wide or tall image;
- photo prompts ("a photo of a cat") describe a different kind of picture.

This module fixes the input side, with no training:

  pixel     find the art's own pixel grid (pixel art is often shipped already
            upscaled, by 2, 3, 4...), shrink back to it, then upscale by a whole
            number with nearest neighbour so every art pixel becomes a clean
            square block; pad to a square with the background colour instead
            of cropping
  cartoon   pad to a square (no crop); no resampling change
  auto      pixel for images that look like pixel art (few colours, long runs
            of identical pixels), cartoon for flat-colour images, unchanged
            otherwise
  off       what the processor does by default

and adds style prompt templates ("pixel art of a {}", "a cartoon of a {}")
to the photo ones, so the text side describes the picture's medium too.

    python scripts/clip_style.py inspect sprite.png --out sprite-224.png
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

STYLES = ("off", "auto", "pixel", "cartoon")
STYLE_TEMPLATES = {
    "pixel": ["pixel art of a {}.", "a 16-bit sprite of a {}.", "a low-resolution pixel art image of a {}.",
              "a retro video game sprite of a {}."],
    "cartoon": ["a cartoon of a {}.", "a flat-colour illustration of a {}.", "a drawing of a {}.",
                "an anime-style picture of a {}."],
}


def _rgb(img):
    return img if img.mode == "RGB" else img.convert("RGB")


def _runs(row) -> list[int]:
    """Lengths of runs of identical values in a sequence."""
    out, n = [], 1
    for a, b in zip(row, row[1:]):
        if a == b:
            n += 1
        else:
            out.append(n)
            n = 1
    out.append(n)
    return out


def block_size(img, max_side: int = 512) -> int:
    """The art's pixel size in screen pixels: the GCD of the run lengths (1 = not upscaled).

    Measured on rows and columns through the image; runs at the edges are ignored, since a crop
    can cut a block."""
    img = _rgb(img)
    w, h = img.size
    if max(w, h) > max_side * 4:
        return 1
    px = img.load()
    lens = Counter()
    for y in range(0, h, max(1, h // 24)):
        r = _runs([px[x, y] for x in range(w)])
        lens.update(r[1:-1])
    for x in range(0, w, max(1, w // 24)):
        r = _runs([px[x, y] for y in range(h)])
        lens.update(r[1:-1])
    if not lens:
        return 1
    # The block is the GCD of the common run lengths; rare odd runs (antialiased edges) are left out.
    total = sum(lens.values())
    common = [n for n, c in lens.items() if c / total >= 0.02]
    g = 0
    for n in common:
        g = math.gcd(g, n)
    return max(1, g)


def looks_like(img) -> str:
    """'pixel', 'cartoon' or 'photo', from colour count and how flat the image is."""
    img = _rgb(img)
    small = img
    if max(img.size) > 256:
        f = 256 / max(img.size)
        small = img.resize((max(1, round(img.size[0] * f)), max(1, round(img.size[1] * f))), resample=0)
    colours = small.getcolors(maxcolors=4096)
    n_col = len(colours) if colours else 4097
    px = small.load()
    w, h = small.size
    same = total = 0
    for y in range(h):
        for x in range(1, w):
            total += 1
            same += px[x, y] == px[x - 1, y]
    flat = same / total if total else 0.0
    if n_col <= 256 and flat >= 0.45 and (block_size(img) > 1 or max(img.size) <= 256):
        return "pixel"
    if n_col <= 1024 and flat >= 0.6:
        return "cartoon"
    return "photo"


def background(img):
    """The most common colour on the border: what to pad with."""
    img = _rgb(img)
    w, h = img.size
    px = img.load()
    edge = [px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)] + \
           [px[0, y] for y in range(h)] + [px[w - 1, y] for y in range(h)]
    return Counter(edge).most_common(1)[0][0]


def pad_square(img, colour=None):
    from PIL import Image
    img = _rgb(img)
    w, h = img.size
    s = max(w, h)
    if w == h:
        return img
    out = Image.new("RGB", (s, s), colour if colour is not None else background(img))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def pixel_upscale(img, target: int = 224):
    """Back to the art's own grid, padded square, then whole-number nearest-neighbour upscale to
    at least `target` (the processor's own resize then only shrinks slightly, or not at all)."""
    from PIL import Image
    img = _rgb(img)
    b = block_size(img)
    if b > 1:
        img = img.resize((max(1, img.size[0] // b), max(1, img.size[1] // b)), Image.NEAREST)
    img = pad_square(img)
    k = max(1, math.ceil(target / img.size[0]))
    out = img.resize((img.size[0] * k, img.size[1] * k), Image.NEAREST)
    if out.size[0] > target and out.size[0] % target:
        # land exactly on the target with nearest too, so the processor does no bicubic smoothing
        out = out.resize((target, target), Image.NEAREST)
    return out


def prepare(img, style: str = "auto", target: int = 224):
    """The image to hand the processor. -> (image, style used)."""
    if style not in STYLES:
        raise ValueError(f"style: one of {', '.join(STYLES)}")
    img = _rgb(img)
    if style == "off":
        return img, "off"
    kind = looks_like(img) if style == "auto" else style
    if kind == "pixel":
        return pixel_upscale(img, target), "pixel"
    if kind == "cartoon":
        return pad_square(img), "cartoon"
    return img, "photo"


def templates(base: list[str], style: str) -> list[str]:
    """Prompt templates: the photo ones plus those for the style (auto and off keep only the base;
    auto adds both style sets, since images of every kind may be in one run)."""
    if style in ("pixel", "cartoon"):
        return base + STYLE_TEMPLATES[style]
    if style == "auto":
        return base + STYLE_TEMPLATES["pixel"] + STYLE_TEMPLATES["cartoon"]
    return list(base)


def target_size(proc) -> int:
    """The processor's input size (224 for most CLIPs, 336, 384 for SigLIP...)."""
    ip = getattr(proc, "image_processor", proc)
    for attr in ("crop_size", "size"):
        v = getattr(ip, attr, None)
        if isinstance(v, dict):
            v = v.get("height") or v.get("shortest_edge") or v.get("width")
        if isinstance(v, int) and v > 0:
            return v
    return 224


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect", help="what auto sees in an image, and the prepared result")
    i.add_argument("image")
    i.add_argument("--style", default="auto", choices=STYLES)
    i.add_argument("--target", type=int, default=224)
    i.add_argument("--out", help="write the prepared image here")
    a = p.parse_args(argv)
    from PIL import Image
    img = Image.open(a.image)
    out, used = prepare(img, a.style, a.target)
    print(f"{a.image}: {img.size[0]}x{img.size[1]}, looks like {looks_like(img)}, block {block_size(img)} px; "
          f"prepared as {used}: {out.size[0]}x{out.size[1]}")
    if a.out:
        out.save(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
