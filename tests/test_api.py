"""REST API used by the React frontend."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from transformers import ViTConfig

from fashion_analyser import api
from fashion_analyser.data import load_metadata
from fashion_analyser.labels import LabelMaps
from fashion_analyser.model import MultiTaskClassifier
from fashion_analyser.train import train


@pytest.fixture(scope="module")
def client(tmp_path_factory, synth_dir):
    cfg = ViTConfig(image_size=224, patch_size=32, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64)
    df = load_metadata(synth_dir, min_per_class=10)
    labels = LabelMaps(sorted(df["category"].unique()))
    model = MultiTaskClassifier.from_backbone_config(cfg, len(labels.categories), len(labels.colours))
    run = tmp_path_factory.mktemp("apirun")
    train(df, run, model=model, epochs=1, freeze_epochs=0, num_workers=0, log=lambda *_: None)
    api.get_analyser.cache_clear()
    mp = pytest.MonkeyPatch()
    mp.setenv("RUN_DIR", str(run))
    yield TestClient(api.app)
    mp.undo()
    api.get_analyser.cache_clear()


def _jpg(h=80, w=60, colour=(0, 0, 200)):
    img = np.full((h, w, 3), 255, np.uint8)
    img[20:60, 15:45] = colour
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and "Shoes" in r["categories"]


def test_analyse_full(client):
    r = client.post("/api/analyse",
                    files={"image": ("p.jpg", _jpg(), "image/jpeg")},
                    data={"title": "Brand Men Red Shoes", "variant": "olive green, suede"})
    assert r.status_code == 200
    body = r.json()
    for key in ["category", "colour", "gender", "conflicts", "warnings", "needs_review",
                "top_categories", "dominant_rgb", "prompt"]:
        assert key in body
    assert body["gender"] == "men"
    assert "olive green, suede" in body["prompt"]["prompt"]


def test_analyse_without_title_or_variant(client):
    r = client.post("/api/analyse", files={"image": ("p.jpg", _jpg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["prompt"] is None


@pytest.mark.parametrize("payload,status", [
    (b"", 422), (b"definitely not an image", 422), (b"x" * (10 * 1024 * 1024 + 1), 413),
])
def test_bad_uploads(client, payload, status):
    r = client.post("/api/analyse", files={"image": ("p.jpg", payload, "image/jpeg")})
    assert r.status_code == status
    assert "detail" in r.json()


def test_overlong_text_rejected(client):
    r = client.post("/api/analyse", files={"image": ("p.jpg", _jpg(), "image/jpeg")},
                    data={"title": "x" * 301})
    assert r.status_code == 422


def test_missing_image_rejected(client):
    assert client.post("/api/analyse", data={"title": "Men Shirt"}).status_code == 422


def test_no_model_is_reported(monkeypatch, tmp_path):
    api.get_analyser.cache_clear()
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "missing"))
    c = TestClient(api.app)
    assert c.get("/api/health").json()["status"] == "no_model"
    r = c.post("/api/analyse", files={"image": ("p.jpg", _jpg(), "image/jpeg")})
    assert r.status_code == 503
    api.get_analyser.cache_clear()


# ---- hosting: model lookup, download and static frontend ---------------------------
def _copy_run(src_client_run, dst):
    import shutil
    shutil.copytree(src_client_run, dst)
    return dst


def test_model_found_in_nested_or_flat_upload(client, tmp_path):
    import os
    import shutil
    run = api.find_run_dir(os.environ["RUN_DIR"])
    # nested: uploaded as an unzipped archive containing the vit/ folder
    nested = tmp_path / "upload" / "vit"
    shutil.copytree(run, nested)
    assert api.find_run_dir(tmp_path / "upload") == nested
    # flat: backbone_config/config.json uploaded at the top level
    flat = tmp_path / "flat"
    shutil.copytree(run, flat)
    shutil.move(str(flat / "backbone_config" / "config.json"), flat / "config.json")
    shutil.rmtree(flat / "backbone_config")
    assert MultiTaskClassifier.load(flat) is not None
    assert api.find_run_dir(tmp_path / "nothing") is None


def test_model_downloaded_from_hub_when_missing(client, tmp_path, monkeypatch):
    import os
    import shutil
    import sys
    import types
    run = api.find_run_dir(os.environ["RUN_DIR"])
    calls = {}

    def fake_snapshot_download(repo_id, local_dir, token=None):
        calls["repo"] = repo_id
        shutil.copytree(run, local_dir, dirs_exist_ok=True)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("MODEL_REPO", "someone/fashion-model")
    assert api.resolve_run_dir() == tmp_path / "model"
    assert calls["repo"] == "someone/fashion-model"
    monkeypatch.setenv("RUN_DIR", os.environ["RUN_DIR"])


def test_static_frontend_served_alongside_api(client, tmp_path, monkeypatch):
    import importlib
    (tmp_path / "index.html").write_text("<html>app</html>")
    monkeypatch.setenv("STATIC_DIR", str(tmp_path))
    mod = importlib.reload(api)
    try:
        c = TestClient(mod.app)
        assert "app" in c.get("/").text
        assert c.get("/api/health").json()["status"] == "ok"
    finally:
        monkeypatch.delenv("STATIC_DIR")
        importlib.reload(api)
