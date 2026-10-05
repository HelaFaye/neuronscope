#!/usr/bin/env python3
"""
Deterministic test images for TestQA's vision subject.

Tasks describe their image as a small spec instead of shipping binary files:

    {"size": [320, 240], "bg": "white",
     "items": [{"shape": "circle", "color": "red", "x": 60, "y": 80, "r": 25},
               {"shape": "square", "color": "blue", "x": 200, "y": 80, "r": 30},
               {"shape": "triangle", "color": "green", "x": 130, "y": 170, "r": 30},
               {"text": "RIVER", "color": "black", "x": 160, "y": 120, "h": 48}]}

`x`, `y` are centres; `r` is the half-width of a shape, `h` the text height.
The same spec always renders the same PNG, so results are comparable across
models and runs.

    python scripts/qa_images.py --bank qa/bank/vision.jsonl --out /tmp/vision-preview
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

COLORS = {"red": (215, 38, 38), "blue": (40, 80, 220), "green": (34, 150, 60), "yellow": (240, 200, 20),
          "black": (20, 20, 20), "white": (255, 255, 255), "orange": (240, 130, 20),
          "purple": (130, 60, 170), "gray": (128, 128, 128)}


def _font(h: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=h)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render(spec: dict) -> bytes:
    from PIL import Image, ImageDraw
    w, h = spec.get("size", [320, 240])
    img = Image.new("RGB", (w, h), COLORS.get(spec.get("bg", "white"), (255, 255, 255)))
    d = ImageDraw.Draw(img)
    for it in spec.get("items", []):
        c = COLORS[it.get("color", "black")]
        x, y = it["x"], it["y"]
        if "text" in it:
            f = _font(int(it.get("h", 40)))
            box = d.textbbox((0, 0), it["text"], font=f)
            d.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), it["text"], fill=c, font=f)
            continue
        r = it.get("r", 20)
        if it["shape"] == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif it["shape"] == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=c)
        elif it["shape"] == "triangle":
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=c)
        else:
            raise ValueError(f"unknown shape {it['shape']!r}")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def data_url(spec_or_path, base: Path | None = None) -> str:
    """PNG data URL for a task's "image" field (a spec dict or a file path)."""
    if isinstance(spec_or_path, dict):
        raw, mime = render(spec_or_path), "image/png"
    else:
        p = Path(spec_or_path)
        if not p.is_absolute() and base is not None:
            p = base / p
        raw = p.read_bytes()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(a.bank).read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            if isinstance(t.get("image"), dict):
                (out / f"{t['id']}.png").write_bytes(render(t["image"]))
    print(f"wrote previews to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
