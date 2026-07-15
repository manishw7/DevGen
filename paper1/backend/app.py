"""
Paper 1 backend — Devanagari handwritten word recognition (TrOCR + LoRA).

Serves every model variant trained for the paper. Interactive Swagger UI at
/docs (OpenAPI at /openapi.json).

Run:
    .venv/bin/python -m uvicorn paper1.backend.app:app --port 8001
"""

from __future__ import annotations

import io
import sys
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.preprocessing import preprocess_pil_for_ocr  # noqa: E402
from paper1.experiments.akshara import split_aksharas  # noqa: E402

BASE_MODEL = "paudelanil/trocr-devanagari-2"

# name -> (adapter_path, full_model_path); paths relative to repo root
MODEL_REGISTRY: "OrderedDict[str, tuple[Optional[str], Optional[str]]]" = OrderedDict([
    ("base", (None, None)),
    ("lora_r4_attn", ("paper1/runs/lora_r4_attn", None)),
    ("lora_r8_attn", ("paper1/runs/lora_r8_attn", None)),
    ("lora_r16_attn", ("paper1/runs/lora_r16_attn", None)),
    ("lora_r32_attn", ("paper1/runs/lora_r32_attn", None)),
    ("lora_r16_attn_ffn", ("paper1/runs/lora_r16_attn_ffn", None)),
    ("lora_r16_legacy", ("paper1/runs/lora_r16_legacy", None)),
    ("lora_r16_legacy_s43", ("paper1/runs/lora_r16_legacy_s43", None)),
    ("lora_r16_legacy_s44", ("paper1/runs/lora_r16_legacy_s44", None)),
    ("full_ft", (None, "paper1/runs/full_ft")),
])
DEFAULT_MODEL = "lora_r16_legacy"
MAX_CACHED_MODELS = 2

# paper numbers (full test split, n=12869) shown in /models
PAPER_METRICS = {
    "base": {"cer": 0.1695, "wacc": 0.6461},
    "lora_r4_attn": {"cer": 0.1141, "wacc": 0.7534},
    "lora_r8_attn": {"cer": 0.1095, "wacc": 0.7614},
    "lora_r16_attn": {"cer": 0.1065, "wacc": 0.7691},
    "lora_r32_attn": {"cer": 0.1030, "wacc": 0.7760},
    "lora_r16_attn_ffn": {"cer": 0.1022, "wacc": 0.7774},
    "lora_r16_legacy": {"cer": 0.1013, "wacc": 0.7786},
    "lora_r16_legacy_s43": {"cer": 0.1017, "wacc": 0.7774},
    "lora_r16_legacy_s44": {"cer": 0.1031, "wacc": 0.7766},
    "full_ft": {"cer": 0.0961, "wacc": 0.7980},
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
_cache: "OrderedDict[str, tuple]" = OrderedDict()  # name -> (processor, model)


def model_available(name: str) -> bool:
    adapter, full = MODEL_REGISTRY[name]
    if adapter:
        return (REPO_ROOT / adapter / "adapter_config.json").exists()
    if full:
        return (REPO_ROOT / full / "model.safetensors").exists()
    return True


def load_model(name: str):
    if name in _cache:
        _cache.move_to_end(name)
        return _cache[name]

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    adapter, full = MODEL_REGISTRY[name]
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    src = str(REPO_ROOT / full) if full else BASE_MODEL
    model = VisionEncoderDecoderModel.from_pretrained(src)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(REPO_ROOT / adapter))
    model = model.to(DEVICE).eval()

    _cache[name] = (processor, model)
    while len(_cache) > MAX_CACHED_MODELS:
        _cache.popitem(last=False)
        if DEVICE == "mps":
            torch.mps.empty_cache()
    return _cache[name]


app = FastAPI(
    title="Paper 1 — Devanagari TrOCR-LoRA Recognition API",
    description=(
        "Serves every model variant from *Parameter-Efficient Adaptation of TrOCR "
        "for Devanagari Handwritten Word Recognition*: the zero-shot base, six LoRA "
        "ablation adapters, two seed repeats, and full fine-tuning.\n\n"
        "Upload a **single cropped word image** to `/recognize` and pick a model "
        "with the `model` query parameter. Paper metrics for each variant are in "
        "`/models`."
    ),
    version="1.0.0",
    contact={"name": "Samir Wagle", "email": "samir@redtab.xyz"},
)


class RecognizeResponse(BaseModel):
    text: str = Field(..., description="Predicted Devanagari word")
    model: str
    confidence: Optional[float] = Field(None, description="Mean per-token probability")
    inference_ms: float
    device: str


class EvaluateRequest(BaseModel):
    prediction: str = Field(..., examples=["अनथथय"])
    reference: str = Field(..., examples=["अनाथों"])


class EvaluateResponse(BaseModel):
    cer: float = Field(..., description="Character error rate (NFC codepoints)")
    aer: float = Field(..., description="Akshara (orthographic syllable) error rate")
    exact_match: bool
    reference_aksharas: list


@app.get("/", tags=["info"], summary="Health check")
def health():
    return {
        "status": "ok",
        "service": "paper1-trocr-backend",
        "device": DEVICE,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_cache.keys()),
        "docs": "/docs",
    }


@app.get("/models", tags=["info"], summary="List model variants + paper metrics")
def list_models():
    return [
        {
            "name": name,
            "available": model_available(name),
            "loaded": name in _cache,
            "adapter_path": MODEL_REGISTRY[name][0],
            "full_model_path": MODEL_REGISTRY[name][1],
            "paper_test_metrics": PAPER_METRICS.get(name),
        }
        for name in MODEL_REGISTRY
    ]


@app.post("/recognize", tags=["recognition"], response_model=RecognizeResponse,
          summary="Recognize one handwritten Devanagari word")
async def recognize(
    file: UploadFile = File(..., description="Image of a SINGLE cropped word (png/jpg)"),
    model: str = Query(DEFAULT_MODEL, enum=list(MODEL_REGISTRY.keys())),
    num_beams: int = Query(4, ge=1, le=8),
    max_length: int = Query(32, ge=4, le=128),
):
    if model not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model '{model}'")
    if not model_available(model):
        raise HTTPException(503, f"Model '{model}' weights not present on disk")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Not a readable image: {exc}")

    processor, net = load_model(model)
    started = time.perf_counter()
    pixel_values = processor(images=preprocess_pil_for_ocr(image), return_tensors="pt").pixel_values.to(DEVICE)

    with torch.inference_mode():
        out = net.generate(
            pixel_values, max_length=max_length, num_beams=num_beams, early_stopping=True,
            decoder_start_token_id=net.config.decoder_start_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    confidence = None
    try:
        scores = net.compute_transition_scores(
            out.sequences, out.scores, beam_indices=getattr(out, "beam_indices", None), normalize_logits=True)
        probs = torch.exp(scores[0])
        if len(probs):
            confidence = round(float(probs.mean()), 4)
    except Exception:
        pass

    return RecognizeResponse(
        text=text, model=model, confidence=confidence,
        inference_ms=round((time.perf_counter() - started) * 1000, 1), device=DEVICE,
    )


@app.post("/evaluate", tags=["evaluation"], response_model=EvaluateResponse,
          summary="CER + akshara error rate between a prediction and reference")
def evaluate(req: EvaluateRequest):
    import editdistance

    pred = unicodedata.normalize("NFC", req.prediction.strip())
    ref = unicodedata.normalize("NFC", req.reference.strip())
    if not ref:
        raise HTTPException(400, "reference must be non-empty")

    cer = editdistance.eval(list(pred), list(ref)) / len(ref)
    pred_ak, ref_ak = split_aksharas(pred), split_aksharas(ref)
    aer = editdistance.eval(pred_ak, ref_ak) / max(1, len(ref_ak))
    return EvaluateResponse(
        cer=round(cer, 4), aer=round(aer, 4),
        exact_match=pred == ref, reference_aksharas=ref_ak,
    )
