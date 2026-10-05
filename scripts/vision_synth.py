#!/usr/bin/env python3
"""
Vision training items whose answers are true by construction.

A text deficit needs a teacher to write a correction and a grader to check it.
For the rendered vision items (qa_images.py) neither is needed: the program
that draws the picture knows how many circles it drew. Each generator below
draws a random scene and returns a TestQA-style task with the answer already
known, in the same skill family as a failed bank item.

    from vision_synth import variations
    for task in variations(bank_task, n=5, seed=0): ...

    python scripts/vision_synth.py --family count_shape -n 4 --out /tmp/vs   # preview

Families are matched to bank items by their prompt, so new bank items in an
existing style get variations automatically; anything unmatched returns none
(and deficits.py says so).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
WORDS = ["RIVER", "STONE", "CLOUD", "MAPLE", "TIGER", "OCEAN", "LEMON", "PIANO", "ROBOT", "CANDLE",
         "BANANA", "SALAD", "APPLE", "PANDA", "DELTA"]
W, H = 320, 240


def _place(rng: random.Random, n: int, r: int, region=(0, 0, W, H), tries: int = 400) -> list[tuple[int, int]]:
    """n non-overlapping centres for shapes of half-width r inside region."""
    x0, y0, x1, y1 = region
    pts: list[tuple[int, int]] = []
    for _ in range(tries):
        if len(pts) == n:
            break
        x, y = rng.randint(x0 + r + 4, x1 - r - 4), rng.randint(y0 + r + 4, y1 - r - 4)
        if all(abs(x - a) > 2 * r + 6 or abs(y - b) > 2 * r + 6 for a, b in pts):
            pts.append((x, y))
    if len(pts) < n:
        raise RuntimeError("could not place shapes")
    return pts


def _task(fid: str, prompt: str, answer: str, answer_type: str, items: list[dict], **extra) -> dict:
    return {"id": fid, "kind": "reasoning", "subject": "vision", "prompt": prompt, "answer": answer,
            "answer_type": answer_type, "image": {"size": [W, H], "bg": "white", "items": items},
            "synthetic": True, **extra}


def count_shape(rng, i):
    shape = rng.choice(SHAPES)
    n = rng.randint(0, 9)
    others = rng.randint(0, 3) if n < 8 else 0
    r = 14
    pts = _place(rng, n + others, r)
    items = [{"shape": shape, "color": rng.choice(COLORS), "x": x, "y": y, "r": r} for x, y in pts[:n]]
    items += [{"shape": rng.choice([s for s in SHAPES if s != shape]), "color": rng.choice(COLORS), "x": x, "y": y,
               "r": r} for x, y in pts[n:]]
    rng.shuffle(items)
    return _task(f"syn-count-{i}", f"How many {shape}s are in this image?", str(n), "number", items)


def count_color(rng, i):
    color = rng.choice(COLORS)
    n, others = rng.randint(0, 6), rng.randint(1, 4)
    pts = _place(rng, n + others, 14)
    items = [{"shape": rng.choice(SHAPES), "color": color, "x": x, "y": y, "r": 14} for x, y in pts[:n]]
    items += [{"shape": rng.choice(SHAPES), "color": rng.choice([c for c in COLORS if c != color]), "x": x, "y": y,
               "r": 14} for x, y in pts[n:]]
    rng.shuffle(items)
    return _task(f"syn-color-count-{i}", f"How many {color} shapes are in this image?", str(n), "number", items)


def count_total(rng, i):
    n = rng.randint(1, 9)
    items = [{"shape": rng.choice(SHAPES), "color": rng.choice(COLORS), "x": x, "y": y, "r": 14}
             for x, y in _place(rng, n, 14)]
    return _task(f"syn-total-{i}", "How many shapes in total are in this image?", str(n), "number", items)


def color_of_shape(rng, i):
    shape, color = rng.choice(SHAPES), rng.choice(COLORS)
    others = [s for s in SHAPES if s != shape]
    pts = _place(rng, 3, 22)
    items = [{"shape": shape, "color": color, "x": pts[0][0], "y": pts[0][1], "r": 22}]
    items += [{"shape": rng.choice(others), "color": rng.choice(COLORS), "x": x, "y": y, "r": 22}
              for x, y in pts[1:1 + rng.randint(0, 2)]]
    return _task(f"syn-color-of-{i}", f"What color is the {shape} in this image? Answer with one word.",
                 color, "text", items)


def shape_of_color(rng, i):
    shape, color = rng.choice(SHAPES), rng.choice(COLORS)
    pts = _place(rng, 3, 22)
    items = [{"shape": shape, "color": color, "x": pts[0][0], "y": pts[0][1], "r": 22}]
    items += [{"shape": rng.choice(SHAPES), "color": rng.choice([c for c in COLORS if c != color]), "x": x, "y": y,
               "r": 22} for x, y in pts[1:1 + rng.randint(0, 2)]]
    return _task(f"syn-shape-of-{i}", f"What shape is the {color} object: circle, square or triangle?",
                 shape, "text", items)


def left_right(rng, i):
    shape, color = rng.choice(SHAPES), rng.choice(COLORS)
    side = rng.choice(["left", "right"])
    x = rng.randint(40, 120) if side == "left" else rng.randint(200, 280)
    items = [{"shape": shape, "color": color, "x": x, "y": rng.randint(50, 190), "r": 24}]
    ox = rng.randint(200, 280) if side == "left" else rng.randint(40, 120)
    items.append({"shape": rng.choice([s for s in SHAPES if s != shape]),
                  "color": rng.choice([c for c in COLORS if c != color]), "x": ox, "y": rng.randint(50, 190), "r": 24})
    return _task(f"syn-lr-{i}", f"Is the {color} {shape} on the left or the right side of the image?",
                 side, "text", items)


def top_bottom(rng, i):
    shape, color = rng.choice(SHAPES), rng.choice(COLORS)
    half = rng.choice(["top", "bottom"])
    y = rng.randint(35, 85) if half == "top" else rng.randint(155, 205)
    items = [{"shape": shape, "color": color, "x": rng.randint(50, 270), "y": y, "r": 24}]
    return _task(f"syn-tb-{i}", f"Is the {color} {shape} in the top half or the bottom half of the image? "
                                "Answer top or bottom.", half, "text", items)


def largest_color(rng, i):
    cols = rng.sample(COLORS, 3)
    sizes = rng.sample([12, 20, 34], 3)
    xs = [60, 160, 260]
    rng.shuffle(xs)
    items = [{"shape": rng.choice(SHAPES), "color": c, "x": x, "y": 120, "r": r} for c, r, x in zip(cols, sizes, xs)]
    big = cols[sizes.index(max(sizes))]
    return _task(f"syn-largest-{i}", "What color is the largest shape in this image? Answer with one word.",
                 big, "text", items)


def ocr_word(rng, i):
    word = rng.choice(WORDS)
    items = [{"text": word, "color": rng.choice(["black", "blue", "red", "purple"]), "x": 160,
              "y": rng.randint(90, 150), "h": 44}]
    return _task(f"syn-ocr-{i}", "What word is written in this image? Answer with the word only.",
                 word, "text", items)


def ocr_number(rng, i):
    num = str(rng.randint(100, 99999))
    items = [{"text": num, "color": "black", "x": 160, "y": 120, "h": 48}]
    return _task(f"syn-ocr-num-{i}", "What number is written in this image?", num, "number", items)


def arith_sum(rng, i):
    a, b = rng.randint(1, 49), rng.randint(1, 49)
    items = [{"text": str(a), "color": "black", "x": 90, "y": 120, "h": 48},
             {"text": str(b), "color": "black", "x": 230, "y": 120, "h": 48}]
    return _task(f"syn-sum-{i}", "The image shows two numbers. What is their sum?", str(a + b), "number", items)


def arith_minus(rng, i):
    a = rng.randint(10, 60)
    b = rng.randint(1, a)
    items = [{"text": f"{a} - {b}", "color": "black", "x": 160, "y": 120, "h": 48}]
    return _task(f"syn-minus-{i}", "The image shows a subtraction. What is the result?", str(a - b), "number", items)


def ocr_color(rng, i):
    color = rng.choice(["red", "blue", "green", "purple", "orange"])
    items = [{"text": rng.choice(WORDS), "color": color, "x": 160, "y": 120, "h": 44}]
    return _task(f"syn-ocr-color-{i}", "What color is the word written in this image? Answer with one word.",
                 color, "text", items)


def letter_count(rng, i):
    word = rng.choice(WORDS)
    letter = rng.choice(sorted(set(word)))
    items = [{"text": word, "color": "black", "x": 160, "y": 120, "h": 44}]
    return _task(f"syn-letters-{i}", f"How many times does the letter {letter} appear in the word written in "
                                     "this image?", str(word.count(letter)), "number", items)


def single_shape(rng, i):
    shape = rng.choice(SHAPES)
    items = [{"shape": shape, "color": rng.choice(COLORS), "x": rng.randint(70, 250), "y": rng.randint(60, 180),
              "r": rng.randint(18, 40)}]
    return _task(f"syn-single-{i}", "This image contains a single shape. Is it a circle, a square or a triangle?",
                 shape, "text", items)


def left_shape(rng, i):
    a, b = rng.sample(SHAPES, 2)
    items = [{"shape": a, "color": rng.choice(COLORS), "x": rng.randint(40, 120), "y": rng.randint(50, 190), "r": 24},
             {"shape": b, "color": rng.choice(COLORS), "x": rng.randint(200, 280), "y": rng.randint(50, 190), "r": 24}]
    return _task(f"syn-left-shape-{i}", "Which shape is on the left side of the image: circle, square or triangle?",
                 a, "text", items)


def larger_choice(rng, i):
    shape = rng.choice(SHAPES)
    ca, cb = rng.sample(COLORS, 2)
    ra, rb = rng.sample([14, 22, 32, 40], 2)
    items = [{"shape": shape, "color": ca, "x": 90, "y": 120, "r": ra},
             {"shape": shape, "color": cb, "x": 230, "y": 120, "r": rb}]
    if rng.random() < 0.5:
        items[0]["x"], items[1]["x"] = 230, 90
    return _task(f"syn-larger-{i}", f"Which is larger in this image, the {ca} {shape} (A) or the {cb} {shape} (B)? "
                                    "Answer A or B.", "A" if ra > rb else "B", "choice", items)


FAMILIES = {
    "single_shape": (single_shape, r"single shape"),
    "left_shape": (left_shape, r"which shape is on the left"),
    "larger_choice": (larger_choice, r"which is larger"),
    "letter_count": (letter_count, r"how many times does the letter"),
    "ocr_color": (ocr_color, r"colou?r is the word"),
    "count_color": (count_color, r"how many (red|blue|green|yellow|orange|purple) shapes"),
    "count_total": (count_total, r"how many shapes in total"),
    "count_shape": (count_shape, r"how many (circle|square|triangle)s"),
    "color_of_shape": (color_of_shape, r"what colou?r is the (circle|square|triangle)"),
    "largest_color": (largest_color, r"colou?r is the largest"),
    "shape_of_color": (shape_of_color, r"what shape is the \w+ object"),
    "left_right": (left_right, r"left or the right"),
    "top_bottom": (top_bottom, r"top half or the bottom half"),
    "ocr_word": (ocr_word, r"what (two )?words? (is|are) written"),
    "ocr_number": (ocr_number, r"what number is written"),
    "arith_sum": (arith_sum, r"two numbers\. what is their sum"),
    "arith_minus": (arith_minus, r"subtraction"),
}


def family_of(task: dict) -> str | None:
    p = task.get("prompt", "").lower()
    for name, (_, pat) in FAMILIES.items():
        if re.search(pat, p):
            return name
    return None


def variations(task: dict, n: int, seed: int = 0) -> list[dict]:
    """n fresh scenes in the failed item's skill family, answers known."""
    fam = family_of(task)
    if fam is None or n <= 0:
        return []
    rng = random.Random(f"{seed}:{task['id']}")
    gen = FAMILIES[fam][0]
    out = []
    for i in range(n):
        t = gen(rng, i)
        t["id"] = f"{task['id']}~v{i}"
        t["family"] = fam
        out.append(t)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from qa_images import render
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.n):
        t = FAMILIES[a.family][0](rng, i)
        (out / f"{t['id']}.png").write_bytes(render(t["image"]))
        print(json.dumps({k: t[k] for k in ("id", "prompt", "answer")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
