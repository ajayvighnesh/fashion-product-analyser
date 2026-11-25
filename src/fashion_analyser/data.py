"""Dataset loading for the Fashion Product Images (Small) dataset.

Expected layout (the Kaggle download, ~44k products):

    data/
      styles.csv        id, gender, masterCategory, subCategory, articleType,
                        baseColour, season, year, usage, productDisplayName
      images/<id>.jpg

`scripts/download_data.py` fetches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .labels import LabelMaps, normalise_colour
from .preprocess import ImageLoadError, augment, load_image, to_model_input


def load_metadata(data_dir: str | Path, top_k_categories: int = 20,
                  min_per_class: int = 50) -> pd.DataFrame:
    """Read styles.csv, drop rows without an image, and keep the main categories.

    styles.csv has a handful of rows with extra commas in the product name,
    so bad lines are skipped rather than crashing the load.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "styles.csv", on_bad_lines="skip")
    df = df.dropna(subset=["id", "articleType"])
    df["id"] = df["id"].astype(int)
    df["image_path"] = df["id"].map(lambda i: str(data_dir / "images" / f"{i}.jpg"))
    df = df[df["image_path"].map(lambda p: Path(p).is_file())]

    df["category"] = df["articleType"].astype(str).str.strip()
    df["colour"] = df["baseColour"].map(normalise_colour)

    counts = df["category"].value_counts()
    keep = counts[counts >= min_per_class].head(top_k_categories).index
    df = df[df["category"].isin(keep)].reset_index(drop=True)
    if df.empty:
        raise ValueError("no usable rows: check the data directory and filters")
    return df


def split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1,
          seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on category."""
    train, rest = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["category"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=test_frac / (val_frac + test_frac),
        stratify=rest["category"], random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class ProductImageDataset(Dataset):
    """Returns (image_tensor, category_idx, colour_idx)."""

    def __init__(self, df: pd.DataFrame, labels: LabelMaps, image_size: int = 224,
                 train: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.image_size = image_size
        self.train = train
        self.rng = np.random.default_rng(seed)
        self._cat = labels.cat2idx
        self._col = labels.col2idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            rgb = load_image(row["image_path"])
        except ImageLoadError:
            # A few dataset files are truncated. A blank image keeps the batch
            # shape intact; such rows are rare enough not to bias training.
            rgb = np.full((self.image_size, self.image_size, 3), 255, np.uint8)
        if self.train:
            rgb = augment(rgb, self.rng)
        x = torch.from_numpy(to_model_input(rgb, self.image_size))
        return x, self._cat[row["category"]], self._col[row["colour"]]


def class_weights(series: pd.Series, classes: list[str]) -> torch.Tensor:
    """Inverse-sqrt-frequency weights, so rare classes are not ignored."""
    counts = series.value_counts().reindex(classes).fillna(0).to_numpy(dtype=np.float32)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)
