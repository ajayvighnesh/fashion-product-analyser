"""REST API used by the React frontend.

    pip install -e ".[api]"
    RUN_DIR=runs/vit uvicorn fashion_analyser.api:app --port 8000

Environment
    RUN_DIR      folder with the trained model (default runs/vit)
    MODEL_REPO   Hugging Face model repo to download the model from when RUN_DIR
                 has no model, e.g. "ajayvighnesh/fashion-product-analyser-vit"
    STATIC_DIR   folder with the built React app; served at / when set
    CORS_ORIGINS comma-separated frontend origins allowed to call the API

Endpoints
    GET  /api/health    model status and label lists
    POST /api/analyse   multipart form: image (file), title (text, optional),
                        variant (text, optional) -> product record + prompt
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agent import analyse_product
from .predict import Analyser
from .preprocess import ImageLoadError
from .prompts import draft_prompt

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 300

app = FastAPI(title="Fashion Product Analyser API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def find_run_dir(root: str | Path) -> Path | None:
    """The folder holding model_meta.json: root itself or any subfolder.

    Lets the model be uploaded as the `vit` folder, its contents, or an unzipped archive.
    """
    root = Path(root)
    if (root / "model_meta.json").is_file():
        return root
    hits = sorted(root.rglob("model_meta.json")) if root.is_dir() else []
    return hits[0].parent if hits else None


def resolve_run_dir() -> Path:
    run_dir = Path(os.environ.get("RUN_DIR", "runs/vit"))
    found = find_run_dir(run_dir)
    if found is None and os.environ.get("MODEL_REPO"):
        from huggingface_hub import snapshot_download
        snapshot_download(os.environ["MODEL_REPO"], local_dir=run_dir,
                          token=os.environ.get("HF_TOKEN") or None)
        found = find_run_dir(run_dir)
    if found is None:
        raise FileNotFoundError(f"no trained model (model_meta.json) under {run_dir}")
    return found


_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load() -> Analyser:
    return Analyser.from_checkpoint(resolve_run_dir(), device=os.environ.get("DEVICE", "cpu"))


def get_analyser() -> Analyser:
    with _load_lock:  # one download/load even if the first requests arrive together
        return _load()


get_analyser.cache_clear = _load.cache_clear  # used by the tests


def _preload() -> None:
    try:
        get_analyser()
    except Exception as e:
        print(f"model not loaded yet: {type(e).__name__}: {e}", flush=True)


if os.environ.get("PRELOAD_MODEL") == "1":
    # Start downloading/loading at boot so the first visitor doesn't wait for it.
    threading.Thread(target=_preload, daemon=True).start()


@app.get("/api/health")
def health():
    try:
        an = get_analyser()
    except Exception as e:  # model missing: report it, keep the server up
        return {"status": "no_model", "detail": f"{type(e).__name__}: {e}"}
    return {"status": "ok", "categories": an.labels.categories, "colours": an.labels.colours}


@app.post("/api/analyse")
async def analyse(image: UploadFile = File(...), title: str = Form(""), variant: str = Form("")):
    if len(title) > MAX_TEXT_CHARS or len(variant) > MAX_TEXT_CHARS:
        raise HTTPException(422, f"Title and design idea must be under {MAX_TEXT_CHARS} characters.")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")

    try:
        an = get_analyser()
    except Exception:
        raise HTTPException(503, "The model isn't loaded on the server yet.")

    try:
        record = analyse_product(an, data, title)
    except ImageLoadError:
        raise HTTPException(422, "That file isn't an image we can read. Try a JPG or PNG.")

    record["prompt"] = (
        draft_prompt(record["category"], record["colour"], record["gender"], variant).to_dict()
        if variant.strip() else None
    )
    return record


# Serve the built React app from the same server (used on Hugging Face Spaces).
# Mounted last so /api/* routes take priority.
_static = os.environ.get("STATIC_DIR")
if _static and Path(_static, "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static, html=True), name="frontend")
