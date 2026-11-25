"""Edge cases for image loading and quality checks."""

import cv2
import numpy as np
import pytest

from fashion_analyser.preprocess import (
    ImageLoadError, augment, check_quality, dominant_colours, load_image, pad_to_square,
    to_model_input,
)


def _png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _textured(h=200, w=200):
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, (h, w, 3), dtype=np.uint8)


def test_grayscale_becomes_rgb():
    gray = np.full((50, 40), 128, np.uint8)
    out = load_image(_png_bytes(gray))
    assert out.shape == (50, 40, 3)


def test_transparent_png_is_composited_on_white():
    bgra = np.zeros((20, 20, 4), np.uint8)  # fully transparent black
    out = load_image(_png_bytes(bgra))
    assert out.min() == 255


def test_bgr_file_is_returned_as_rgb(tmp_path):
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[:, :, 2] = 255  # pure red in BGR order
    p = tmp_path / "red.png"
    cv2.imwrite(str(p), bgr)
    out = load_image(p)
    assert tuple(out[0, 0]) == (255, 0, 0)


def test_16bit_png_is_converted():
    img16 = (np.arange(100, dtype=np.uint16).reshape(10, 10) * 600)
    out = load_image(_png_bytes(img16))
    assert out.dtype == np.uint8 and out.shape == (10, 10, 3)


@pytest.mark.parametrize("bad", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_corrupt_bytes_raise(bad):
    with pytest.raises(ImageLoadError):
        load_image(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.jpg")


def test_text_file_with_jpg_extension_raises(tmp_path):
    p = tmp_path / "fake.jpg"
    p.write_text("hello")
    with pytest.raises(ImageLoadError):
        load_image(p)


def test_good_image_has_no_warnings():
    assert check_quality(_textured()).ok


def test_tiny_image_warns():
    assert any("very small" in w for w in check_quality(_textured(20, 20)).warnings)


def test_extreme_aspect_warns():
    assert any("aspect" in w for w in check_quality(_textured(50, 400)).warnings)


def test_blurry_image_warns():
    blurred = cv2.GaussianBlur(_textured(), (31, 31), 15)
    assert any("blurry" in w for w in check_quality(blurred).warnings)


def test_dark_image_warns():
    dark = (_textured() // 8).astype(np.uint8)
    assert any("dark" in w for w in check_quality(dark).warnings)


def test_blank_white_image_warns():
    assert not check_quality(np.full((100, 100, 3), 255, np.uint8)).ok


def test_pad_to_square_keeps_content_centered():
    img = np.zeros((10, 30, 3), np.uint8)
    sq = pad_to_square(img)
    assert sq.shape == (30, 30, 3)
    assert sq[0, 0, 0] == 255 and sq[15, 15, 0] == 0


@pytest.mark.parametrize("shape", [(80, 60, 3), (1000, 50, 3), (5, 5, 3)])
def test_model_input_shape_and_dtype(shape):
    x = to_model_input(np.full(shape, 100, np.uint8), size=224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_augment_never_changes_hue():
    img = np.zeros((60, 60, 3), np.uint8)
    img[:, :] = (200, 30, 30)
    rng = np.random.default_rng(1)
    for _ in range(20):
        out = augment(img, rng)
        r, g, b = out.reshape(-1, 3).mean(0)
        assert r > 150 and g < 50 and b < 50


def test_dominant_colour_ignores_white_background():
    img = np.full((80, 60, 3), 255, np.uint8)
    img[20:60, 15:45] = (0, 0, 200)
    (rgb, share), *_ = dominant_colours(img, k=2)
    assert rgb[2] > 150 and rgb[0] < 50 and share > 0.5
