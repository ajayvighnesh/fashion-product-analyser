"""Create a tiny synthetic dataset in the real dataset's format.

Used by the tests and for a smoke run of the whole pipeline without
downloading anything: each "product" is a simple shape (category) drawn in a
solid colour (colour label) on a white background.

    python scripts/make_synthetic_data.py --out data_synth --per-class 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLOURS_BGR = {"Navy Blue": (128, 0, 0), "Red": (0, 0, 220), "Black": (20, 20, 20),
               "Green": (0, 160, 0), "Yellow": (0, 220, 240)}
SHAPES = ["Tshirts", "Shoes", "Watches"]


def draw(shape: str, colour, rng) -> np.ndarray:
    img = np.full((80, 60, 3), 255, np.uint8)
    jitter = lambda v: int(v + rng.integers(-4, 5))  # noqa: E731
    if shape == "Tshirts":
        pts = np.array([[jitter(10), 15], [jitter(22), 8], [jitter(38), 8], [jitter(50), 15],
                        [44, 28], [40, 24], [40, 70], [20, 70], [20, 24], [16, 28]], np.int32)
        cv2.fillPoly(img, [pts], colour)
    elif shape == "Shoes":
        cv2.ellipse(img, (30, jitter(52)), (jitter(24), 12), 0, 0, 360, colour, -1)
        cv2.rectangle(img, (8, 52), (52, 60), colour, -1)
    else:
        cv2.circle(img, (30, jitter(40)), jitter(14), colour, -1)
        cv2.rectangle(img, (25, 5), (35, 75), colour, -1)
    return img


def make(out: str | Path, per_class: int = 40, seed: int = 0) -> Path:
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, pid = [], 1000
    for shape in SHAPES:
        for _ in range(per_class):
            cname = list(COLOURS_BGR)[int(rng.integers(len(COLOURS_BGR)))]
            cv2.imwrite(str(out / "images" / f"{pid}.jpg"), draw(shape, COLOURS_BGR[cname], rng))
            gender = ["Men", "Women"][int(rng.integers(2))]
            rows.append({"id": pid, "gender": gender, "masterCategory": "Apparel",
                         "subCategory": "x", "articleType": shape, "baseColour": cname,
                         "season": "Summer", "year": 2020, "usage": "Casual",
                         "productDisplayName": f"Brand {gender} {cname} {shape}"})
            pid += 1
    pd.DataFrame(rows).to_csv(out / "styles.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_synth")
    p.add_argument("--per-class", type=int, default=40)
    a = p.parse_args()
    print(make(a.out, a.per_class))
