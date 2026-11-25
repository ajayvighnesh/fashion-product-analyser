"""Fine-tune the multi-task classifier.

    python -m fashion_analyser.train --data-dir data --out-dir runs/vit --epochs 5

Phase 1 trains only the heads with the backbone frozen (fast, stable);
phase 2 unfreezes everything at a lower learning rate. The best epoch by
validation macro-F1 is kept, then evaluated once on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ProductImageDataset, class_weights, load_metadata, split
from .labels import LabelMaps
from .model import DEFAULT_BACKBONE, MultiTaskClassifier


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cat_true, cat_pred, col_true, col_pred = [], [], [], []
    for x, yc, yl in loader:
        lc, ll = model(x.to(device))
        cat_pred += lc.argmax(1).cpu().tolist()
        col_pred += ll.argmax(1).cpu().tolist()
        cat_true += yc.tolist()
        col_true += yl.tolist()
    return {
        "category_acc": accuracy_score(cat_true, cat_pred),
        "category_macro_f1": f1_score(cat_true, cat_pred, average="macro", zero_division=0),
        "colour_acc": accuracy_score(col_true, col_pred),
        "colour_macro_f1": f1_score(col_true, col_pred, average="macro", zero_division=0),
        "_raw": (cat_true, cat_pred, col_true, col_pred),
    }


def _public(m: dict) -> dict:
    return {k: round(float(v), 4) for k, v in m.items() if not k.startswith("_")}


def train(df: pd.DataFrame, out_dir: str | Path, model: MultiTaskClassifier | None = None,
          backbone: str = DEFAULT_BACKBONE, epochs: int = 5, freeze_epochs: int = 1,
          batch_size: int = 32, lr_head: float = 1e-3, lr_backbone: float = 3e-5,
          image_size: int = 224, num_workers: int = 2, device: str = "auto",
          seed: int = 42, log=print) -> dict:
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    train_df, val_df, test_df = split(df, seed=seed)
    labels = LabelMaps(categories=sorted(df["category"].unique()))
    labels.save(out_dir / "labels.json")
    log(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)} | "
        f"{len(labels.categories)} categories | {len(labels.colours)} colours | device {dev}")

    mk = lambda d, tr: DataLoader(  # noqa: E731
        ProductImageDataset(d, labels, image_size, train=tr, seed=seed),
        batch_size=batch_size, shuffle=tr, num_workers=num_workers,
    )
    train_dl, val_dl, test_dl = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    if model is None:
        model = MultiTaskClassifier.from_pretrained_backbone(
            len(labels.categories), len(labels.colours), backbone)
    model.to(dev)

    loss_cat = nn.CrossEntropyLoss(
        weight=class_weights(train_df["category"], labels.categories).to(dev), label_smoothing=0.1)
    loss_col = nn.CrossEntropyLoss(
        weight=class_weights(train_df["colour"], labels.colours).to(dev), label_smoothing=0.1)

    history, best_score, best_epoch = [], -1.0, -1
    for epoch in range(1, epochs + 1):
        frozen = epoch <= freeze_epochs
        model.set_backbone_trainable(not frozen)
        params = [
            {"params": list(model.category_head.parameters()) + list(model.colour_head.parameters()),
             "lr": lr_head},
        ]
        if not frozen:
            params.append({"params": model.backbone.parameters(), "lr": lr_backbone})
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in params], total_steps=max(1, len(train_dl)))

        model.train()
        t0, running = time.time(), 0.0
        for x, yc, yl in train_dl:
            x, yc, yl = x.to(dev), yc.to(dev), yl.to(dev)
            lc, ll = model(x)
            loss = loss_cat(lc, yc) + loss_col(ll, yl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(x)

        val = evaluate(model, val_dl, dev)
        score = (val["category_macro_f1"] + val["colour_macro_f1"]) / 2
        row = {"epoch": epoch, "backbone_frozen": frozen,
               "train_loss": round(running / len(train_dl.dataset), 4),
               **{f"val_{k}": v for k, v in _public(val).items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log(json.dumps(row))
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save(out_dir)

    best = MultiTaskClassifier.load(out_dir).to(dev)
    test = evaluate(best, test_dl, dev)
    ct, cp, lt, lp = test["_raw"]
    pd.DataFrame(confusion_matrix(ct, cp, labels=range(len(labels.categories))),
                 index=labels.categories, columns=labels.categories).to_csv(
        out_dir / "confusion_category.csv")
    pd.DataFrame(confusion_matrix(lt, lp, labels=range(len(labels.colours))),
                 index=labels.colours, columns=labels.colours).to_csv(
        out_dir / "confusion_colour.csv")

    metrics = {"backbone": best.backbone_name, "best_epoch": best_epoch,
               "test": _public(test), "history": history,
               "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"best epoch {best_epoch} | test {json.dumps(_public(test))}")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--top-k-categories", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None,
                   help="subsample for a quick run on a laptop CPU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    df = load_metadata(a.data_dir, top_k_categories=a.top_k_categories)
    if a.max_samples and a.max_samples < len(df):
        # stratified subsample: same share of every category
        df = df.groupby("category").sample(frac=a.max_samples / len(df),
                                           random_state=a.seed).reset_index(drop=True)
    train(df, a.out_dir, backbone=a.backbone, epochs=a.epochs, freeze_epochs=a.freeze_epochs,
          batch_size=a.batch_size, image_size=a.image_size, num_workers=a.num_workers,
          device=a.device, seed=a.seed)


if __name__ == "__main__":
    main()
