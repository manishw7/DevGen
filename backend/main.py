"""
DevGen Framework — FastAPI Backend
Neuro-Generative Suite for Devanagari Handwritten Text Recognition

Endpoints:
  GET  /                          → Health check
  GET  /api/v1/dataset/info       → Dataset metadata
  GET  /api/v1/dataset/sample     → Single sample
  GET  /api/v1/dataset/random     → Random samples
  GET  /api/v1/dataset/browse     → Paginated samples
  POST /api/v1/recognize          → OCR recognition
  POST /api/v1/recognize/full     → OCR + NER in one shot
  POST /api/v1/preprocess         → Preprocess an image
  POST /api/v1/evaluate           → CER evaluation
  POST /api/v1/ner                → NER extraction from plain text
"""

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import io
import os
import traceback

from backend.preprocessing import full_preprocess, preprocess_for_ocr
from backend.dataset_loader import get_dataset_info, get_sample, get_random_samples, get_paginated_samples
from backend.ner_extractor import extract_entities, summarize_entities
from backend.trocr_engine import CERCalculator, TrOCREngine, inspect_model_artifacts

# Lazy-load heavy ML models
_trocr_engine = None


def get_trocr_engine():
    """Lazy-load the TrOCR engine to avoid long startup times."""
    global _trocr_engine
    if _trocr_engine is None:
        _trocr_engine = TrOCREngine()
    return _trocr_engine


app = FastAPI(
    title="DevGen Framework API",
    description="Neuro-Generative Suite for Devanagari Handwritten Text Recognition",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    model_info = inspect_model_artifacts()
    return {
        "status": "ok",
        "service": "DevGen Framework",
        "version": "0.1.0",
        "modules": ["preprocessing", "trocr_engine", "dataset_loader", "ner_extractor"],
        "model_loaded": _trocr_engine is not None,
        "model": {
            "base_model_name": model_info["base_model_name"],
            "adapter_path": model_info["adapter_path"],
            "adapter_source": model_info["adapter_source"],
            "device": model_info["device"],
            "using_adapter": model_info["using_adapter"],
        },
    }


@app.get("/api/v1/model/info")
def model_info():
    """Return model configuration and checkpoint discovery status."""
    try:
        if _trocr_engine is not None:
            return {"status": "ok", "data": _trocr_engine.info()}

        discovered = inspect_model_artifacts()
        discovered["loaded"] = False
        discovered["default_max_length"] = int(os.getenv("TROCR_MAX_LENGTH", "128"))
        discovered["num_beams"] = 4
        return {"status": "ok", "data": discovered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset Endpoints ──────────────────────────────────────────────────────

@app.get("/api/v1/dataset/info")
def dataset_info():
    """Get dataset metadata (splits, sizes, columns)."""
    try:
        info = get_dataset_info()
        return {"status": "ok", "data": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/sample")
def dataset_sample(
    split: str = Query("train", description="Dataset split: train, validation, test"),
    index: int = Query(0, description="Sample index"),
):
    """Get a single dataset sample by index."""
    try:
        sample = get_sample(split, index)
        return {"status": "ok", "data": sample}
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/random")
def dataset_random(
    split: str = Query("train", description="Dataset split"),
    count: int = Query(12, description="Number of random samples"),
):
    """Get random samples from the dataset."""
    try:
        samples = get_random_samples(split, count)
        return {"status": "ok", "data": samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dataset/browse")
def dataset_browse(
    split: str = Query("train", description="Dataset split"),
    page: int = Query(0, description="Page number (0-indexed)"),
    page_size: int = Query(20, description="Samples per page"),
):
    """Paginated dataset browsing."""
    try:
        result = get_paginated_samples(split, page, page_size)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Preprocessing Endpoint ─────────────────────────────────────────────────

@app.post("/api/v1/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """Preprocess a document image (denoise, deskew, normalize).
    Returns the preprocessed image as base64."""
    try:
        contents = await file.read()
        processed_pil = full_preprocess(contents)

        buffer = io.BytesIO()
        processed_pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "ok",
            "filename": file.filename,
            "preprocessed_image_base64": b64,
            "size": {"width": processed_pil.width, "height": processed_pil.height},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Recognition Endpoint ───────────────────────────────────────────────────

@app.post("/api/v1/recognize")
async def recognize_document(file: UploadFile = File(...)):
    """Full OCR pipeline: Preprocess → TrOCR Recognition.
    Returns recognized text with per-token confidence scores."""
    try:
        contents = await file.read()
        
        # Step 1: Crop/prepare for the word-level TrOCR model.
        processed_image = preprocess_for_ocr(contents)
        
        # Step 2: TrOCR Recognition
        engine = get_trocr_engine()
        result = engine.recognize(processed_image)
        
        # Include preprocessed image for frontend heatmap overlay
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        result["preprocessed_image_base64"] = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")
        result["filename"] = file.filename
        result["status"] = "ok"
        
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation Endpoint ────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    prediction: str
    reference: str


@app.post("/api/v1/evaluate")
def evaluate_cer(req: EvalRequest):
    """Calculate Character Error Rate between prediction and reference."""
    try:
        result = CERCalculator.calculate(req.prediction, req.reference)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── NER Endpoint ───────────────────────────────────────────────────────────

class NERRequest(BaseModel):
    text: str


@app.post("/api/v1/ner")
def extract_ner(req: NERRequest):
    """Extract structured entities (dates, IDs, districts) from recognized text."""
    try:
        entities = extract_entities(req.text)
        summary = summarize_entities(entities)
        return {"status": "ok", "data": {"entities": entities, "summary": summary}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recognize/full")
async def recognize_and_extract(file: UploadFile = File(...)):
    """Full pipeline: Preprocess → TrOCR Recognition → NER Extraction.
    Returns recognized text, confidence scores, AND structured entities."""
    try:
        contents = await file.read()

        # Step 1: Crop/prepare for the word-level TrOCR model.
        processed_image = preprocess_for_ocr(contents)

        # Step 2: TrOCR Recognition
        engine = get_trocr_engine()
        result = engine.recognize(processed_image)

        # Step 3: NER Extraction
        entities = extract_entities(result["text"])
        result["entities"] = entities
        result["entities_summary"] = summarize_entities(entities)

        # Include preprocessed image
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        result["preprocessed_image_base64"] = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")
        result["filename"] = file.filename
        result["status"] = "ok"

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
