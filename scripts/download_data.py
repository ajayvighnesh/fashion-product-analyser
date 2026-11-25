"""Download the Fashion Product Images (Small) dataset into data/.

Two sources, same data (~44k product photos, 80x60px, plus metadata):

    python scripts/download_data.py                 # Hugging Face (no account needed)
    python scripts/download_data.py --source kaggle # Kaggle (needs a Kaggle API token)

Result: data/styles.csv and data/images/<id>.jpg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HF_DATASET = "ashraq/fashion-product-images-small"
KAGGLE_DATASET = "paramaggarwal/fashion-product-images-small"
COLUMNS = ["id", "gender", "masterCategory", "subCategory", "articleType",
           "baseColour", "season", "year", "usage", "productDisplayName"]


def from_hf(out: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for ex in ds:
        ex["image"].convert("RGB").save(out / "images" / f"{ex['id']}.jpg", quality=95)
        rows.append({c: ex.get(c) for c in COLUMNS})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "styles.csv", index=False)


def from_kaggle(out: Path) -> None:
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    root = next(p.parent for p in src.rglob("styles.csv"))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "styles.csv", out / "styles.csv")
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    shutil.copytree(root / "images", out / "images")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    out = Path(a.out)
    (from_hf if a.source == "hf" else from_kaggle)(out)
    print(f"done: {len(list((out / 'images').glob('*.jpg')))} images in {out}")
