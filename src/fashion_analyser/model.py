"""Multi-task classifier: one pretrained Hugging Face backbone, two heads.

Category and colour share the visual features; each head is a small MLP.
Any Hugging Face vision backbone that exposes `pooler_output` (ViT, ConvNeXt,
Swin, ResNet, ...) works.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

DEFAULT_BACKBONE = "google/vit-base-patch16-224-in21k"


def _hidden_size(config) -> int:
    for attr in ("hidden_size", "hidden_sizes"):
        v = getattr(config, attr, None)
        if v is not None:
            return v[-1] if isinstance(v, (list, tuple)) else v
    raise ValueError("cannot infer backbone feature size")


class Head(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_categories: int, n_colours: int,
                 backbone_name: str = "custom"):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        dim = _hidden_size(backbone.config)
        self.category_head = Head(dim, n_categories)
        self.colour_head = Head(dim, n_colours)

    @classmethod
    def from_pretrained_backbone(cls, n_categories: int, n_colours: int,
                                 backbone_name: str = DEFAULT_BACKBONE):
        backbone = AutoModel.from_pretrained(backbone_name)
        return cls(backbone, n_categories, n_colours, backbone_name)

    @classmethod
    def from_backbone_config(cls, config, n_categories: int, n_colours: int):
        """Randomly initialised backbone (used by the tests, no download)."""
        return cls(AutoModel.from_config(config), n_categories, n_colours, "from-config")

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:  # e.g. ViT built without a pooling layer
            pooled = out.last_hidden_state[:, 0]
        return pooled.flatten(1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(pixel_values)
        return self.category_head(f), self.colour_head(f)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    # -- checkpointing ---------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.config.save_pretrained(out_dir / "backbone_config")
        torch.save(self.state_dict(), out_dir / "model.pt")
        (out_dir / "model_meta.json").write_text(json.dumps({
            "backbone_name": self.backbone_name,
            "n_categories": self.category_head.net[-1].out_features,
            "n_colours": self.colour_head.net[-1].out_features,
        }, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path, map_location: str = "cpu") -> "MultiTaskClassifier":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "model_meta.json").read_text())
        config = AutoConfig.from_pretrained(out_dir / "backbone_config")
        model = cls(AutoModel.from_config(config), meta["n_categories"], meta["n_colours"],
                    meta["backbone_name"])
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location=map_location))
        model.eval()
        return model
