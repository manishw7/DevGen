"""
Paper 2 backend — Devanagari handwriting generation (from-scratch LDM).

Generates 64x256 handwritten word images from Devanagari text using the
trained latent diffusion models (akshara / codepoint conditioning).
Interactive Swagger UI at /docs.

Run:
    .venv/bin/python -m uvicorn paper2.backend.app:app --port 8002
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINTS = {
    "akshara": REPO_ROOT / "paper2/runs/ldm_akshara/checkpoint.pt",
    "codepoint": REPO_ROOT / "paper2/runs/ldm_codepoint/checkpoint.pt",
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
_models: dict = {}  # mode -> (unet, encoder, tokenizer, epoch)
_vae = None


def load_mode(mode: str):
    global _vae
    if mode in _models:
        return _models[mode]
    from paper2.ldm.sample_ldm import load_model
    path = CHECKPOINTS[mode]
    if not path.exists():
        raise HTTPException(503, f"No checkpoint for '{mode}' at {path}")
    _models[mode] = load_model(str(path), DEVICE)
    return _models[mode]


app = FastAPI(
    title="Paper 2 — Devanagari Handwriting LDM API",
    description=(
        "Text-to-handwriting generation for Devanagari using a compact latent "
        "diffusion model trained from scratch (companion paper to the TrOCR-LoRA "
        "recognition study). Two conditioning variants: **akshara** (orthographic-"
        "syllable tokens) and **codepoint** (Unicode codepoint tokens).\n\n"
        "POST a Devanagari word to `/generate` — returns a PNG. "
        "`/generate/b64` returns JSON with base64 images for batch use."
    ),
    version="1.0.0",
    contact={"name": "Samir Wagle", "email": "samir@redtab.xyz"},
)


class GenerateRequest(BaseModel):
    text: str = Field(..., examples=["नमस्ते"], description="Devanagari word to render")
    mode: str = Field("akshara", pattern="^(akshara|codepoint)$")
    steps: int = Field(50, ge=10, le=200, description="DDIM sampling steps")
    guidance: float = Field(1.5, ge=0.0, le=8.0, description="CFG scale (1.0-1.5 works best)")
    seed: int = Field(42, description="Random seed — vary for different handwriting")
    count: int = Field(1, ge=1, le=8, description="Number of samples (distinct seeds)")


@app.get("/", tags=["info"], summary="Health check")
def health():
    return {
        "status": "ok",
        "service": "paper2-ldm-backend",
        "device": DEVICE,
        "checkpoints": {
            m: {"present": p.exists(), "path": str(p.relative_to(REPO_ROOT))}
            for m, p in CHECKPOINTS.items()
        },
        "loaded": {m: f"epoch {v[3]}" for m, v in _models.items()},
        "docs": "/docs",
    }


@app.post("/generate", tags=["generation"], summary="Generate one handwriting image (PNG)",
          response_class=Response, responses={200: {"content": {"image/png": {}}}})
def generate(req: GenerateRequest):
    from paper2.ldm.sample_ldm import sample_batch
    global _vae

    unet, encoder, tokenizer, epoch = load_mode(req.mode)
    started = time.perf_counter()
    images, _vae = sample_batch(
        unet, encoder, tokenizer, [req.text], DEVICE, vae=_vae,
        steps=req.steps, guidance=req.guidance, seed=req.seed,
    )
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    ms = round((time.perf_counter() - started) * 1000)
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Model-Epoch": str(epoch), "X-Inference-Ms": str(ms)})


@app.post("/generate/b64", tags=["generation"], summary="Generate images, JSON + base64")
def generate_b64(req: GenerateRequest):
    from paper2.ldm.sample_ldm import sample_batch
    global _vae

    unet, encoder, tokenizer, epoch = load_mode(req.mode)
    started = time.perf_counter()
    out = []
    for i in range(req.count):
        images, _vae = sample_batch(
            unet, encoder, tokenizer, [req.text], DEVICE, vae=_vae,
            steps=req.steps, guidance=req.guidance, seed=req.seed + i,
        )
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        out.append({"seed": req.seed + i, "png_base64": base64.b64encode(buf.getvalue()).decode()})
    return {
        "text": req.text, "mode": req.mode, "model_epoch": epoch,
        "steps": req.steps, "guidance": req.guidance,
        "inference_ms": round((time.perf_counter() - started) * 1000),
        "images": out,
    }
