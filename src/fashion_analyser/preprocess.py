"""OpenCV image preprocessing and quality checks.

Product photos in the wild are messy: grayscale scans, PNGs with transparent
backgrounds, very wide banners, tiny thumbnails, blurry phone shots or files
that are not images at all. Everything is normalised here before it reaches
the model, and quality problems are reported instead of silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MIN_SIDE = 32  # below this the image carries too little detail to classify
BLUR_THRESHOLD = 60.0  # variance of the Laplacian; lower means blurrier
DARK_THRESHOLD = 35.0
BRIGHT_THRESHOLD = 245.0
MAX_ASPECT = 3.0


class ImageLoadError(ValueError):
    """Raised when a file cannot be decoded as an image."""


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    brightness: float
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def load_image(source: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Load an image as an RGB uint8 array, whatever its original format.

    Handles grayscale, BGRA/transparent PNGs (composited onto white) and
    raw bytes. Raises ImageLoadError for unreadable input.
    """
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 3 and img.shape[2] == 3:
            return img.astype(np.uint8)  # assume already RGB
    elif isinstance(source, (bytes, bytearray)):
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None
    else:
        path = Path(source)
        if not path.is_file():
            raise ImageLoadError(f"file not found: {path}")
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if img is None or img.size == 0:
        raise ImageLoadError("could not decode image (corrupt or unsupported format)")

    if img.dtype != np.uint8:  # e.g. 16-bit PNG
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = bgr * alpha + 255.0 * (1.0 - alpha)  # composite onto white
        return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def check_quality(rgb: np.ndarray) -> QualityReport:
    """Measure size, sharpness and exposure and list any problems."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    warnings: list[str] = []
    if min(h, w) < MIN_SIDE:
        warnings.append(f"image is very small ({w}x{h}px)")
    if max(h, w) / max(1, min(h, w)) > MAX_ASPECT:
        warnings.append(f"extreme aspect ratio ({w}x{h}); product may be cropped or tiny")
    if blur < BLUR_THRESHOLD:
        warnings.append(f"image looks blurry (sharpness {blur:.0f})")
    if brightness < DARK_THRESHOLD:
        warnings.append("image is very dark; colour prediction is unreliable")
    elif brightness > BRIGHT_THRESHOLD:
        warnings.append("image is almost entirely white/overexposed")
    return QualityReport(w, h, blur, brightness, warnings)


def pad_to_square(rgb: np.ndarray, fill: int = 255) -> np.ndarray:
    """Pad with a white border so the product is not distorted by resizing."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(
        rgb, top, side - h - top, left, side - w - left,
        cv2.BORDER_CONSTANT, value=(fill, fill, fill),
    )


def to_model_input(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Square-pad, resize and ImageNet-normalise. Returns a CHW float32 array."""
    square = pad_to_square(rgb)
    interp = cv2.INTER_AREA if square.shape[0] > size else cv2.INTER_CUBIC
    resized = cv2.resize(square, (size, size), interpolation=interp)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light training augmentation. Hue is never changed, since colour is a label."""
    out = rgb
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        scale = rng.uniform(0.85, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y = int(rng.integers(0, h - ch + 1))
        x = int(rng.integers(0, w - cw + 1))
        out = out[y : y + ch, x : x + cw]
    if rng.random() < 0.3:
        factor = rng.uniform(0.9, 1.1)  # small brightness change only
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def dominant_colours(rgb: np.ndarray, k: int = 3) -> list[tuple[tuple[int, int, int], float]]:
    """K-means on non-background pixels: returns [(rgb, share), ...].

    Used as a cross-check for the colour head and shown in the demo.
    """
    small = cv2.resize(pad_to_square(rgb), (96, 96), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    not_bg = ~np.all(pixels > 235, axis=1)  # drop the near-white studio background
    pixels = pixels[not_bg] if not_bg.sum() >= k * 10 else pixels
    k = max(1, min(k, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(-counts)
    return [
        (tuple(int(c) for c in centres[i]), float(counts[i] / counts.sum())) for i in order
    ]
