"""REST API used by the React frontend.

    pip install -e ".[api]"
    RUN_DIR=runs/vit uvicorn fashion_analyser.api:app --port 8000

Endpoints
    GET  /api/health    model status and label lists
    POST /api/analyse   multipart form: image (file), title (text, optional),
                        variant (text, optional) -> product record + prompt
"""

from __future__ import annotations

import os
from functools import lru_cache

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


@lru_cache(maxsize=1)
def get_analyser() -> Analyser:
    run_dir = os.environ.get("RUN_DIR", "runs/vit")
    device = os.environ.get("DEVICE", "cpu")
    return Analyser.from_checkpoint(run_dir, device=device)


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
