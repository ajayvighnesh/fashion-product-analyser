"""Inference: image in, structured product attributes out."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .labels import LabelMaps
from .model import MultiTaskClassifier
from .preprocess import check_quality, dominant_colours, load_image, to_model_input

LOW_CONFIDENCE = 0.5


@dataclass
class Prediction:
    category: str
    category_confidence: float
    colour: str
    colour_confidence: float
    top_categories: list[tuple[str, float]]
    top_colours: list[tuple[str, float]]
    dominant_rgb: list[tuple[tuple[int, int, int], float]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Analyser:
    def __init__(self, model: MultiTaskClassifier, labels: LabelMaps,
                 image_size: int = 224, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.labels = labels
        self.image_size = image_size
        self.device = device

    @classmethod
    def from_checkpoint(cls, run_dir: str | Path, device: str = "cpu") -> "Analyser":
        run_dir = Path(run_dir)
        return cls(MultiTaskClassifier.load(run_dir, device),
                   LabelMaps.load(run_dir / "labels.json"), device=device)

    @torch.no_grad()
    def predict(self, image, top_k: int = 3) -> Prediction:
        rgb = load_image(image)
        quality = check_quality(rgb)
        x = torch.from_numpy(to_model_input(rgb, self.image_size)).unsqueeze(0).to(self.device)
        lc, ll = self.model(x)
        pc = torch.softmax(lc, 1)[0].cpu().numpy()
        pl = torch.softmax(ll, 1)[0].cpu().numpy()

        def top(p: np.ndarray, names: list[str]):
            idx = np.argsort(-p)[:top_k]
            return [(names[i], round(float(p[i]), 4)) for i in idx]

        tc, tl = top(pc, self.labels.categories), top(pl, self.labels.colours)
        warnings = list(quality.warnings)
        if tc[0][1] < LOW_CONFIDENCE:
            warnings.append(f"low confidence on category ({tc[0][1]:.0%}); "
                            f"could also be {tc[1][0]}" if len(tc) > 1 else "low confidence on category")
        if tl[0][1] < LOW_CONFIDENCE:
            warnings.append(f"low confidence on colour ({tl[0][1]:.0%})")
        return Prediction(
            category=tc[0][0], category_confidence=tc[0][1],
            colour=tl[0][0], colour_confidence=tl[0][1],
            top_categories=tc, top_colours=tl,
            dominant_rgb=dominant_colours(rgb), warnings=warnings,
        )
