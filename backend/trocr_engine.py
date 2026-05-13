"""
DevGen Framework — TrOCR Recognition Engine
Loads a base TrOCR model with an optional LoRA adapter for Devanagari OCR.
Also integrates a CNN character classifier for single-character inputs.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

DEFAULT_BASE_MODEL = "microsoft/trocr-base-handwritten"
DEFAULT_ADAPTER_ROOT = "trocr-devanagari-lora"
DEFAULT_IMAGE_PROCESSOR = "google/vit-base-patch16-224-in21k"
SPECIAL_TOKEN_NAMES = ("bos_token_id", "cls_token_id", "eos_token_id", "pad_token_id", "sep_token_id")


def get_best_torch_device() -> str:
    """Return the best available PyTorch device for this machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
            return "mps"
        except RuntimeError as exc:
            print(f"[TrOCR Engine] MPS unavailable at runtime, falling back to CPU: {exc}")
    return "cpu"


def load_trocr_processor(model_name: str, image_processor_source: Optional[str] = None) -> TrOCRProcessor:
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except Exception as exc:
        print(f"[TrOCR Engine] Processor fallback for '{model_name}': {exc}")
        image_processor = ViTImageProcessor.from_pretrained(image_processor_source or DEFAULT_IMAGE_PROCESSOR)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


@dataclass
class ModelArtifacts:
    project_root: str
    base_model_name: str
    adapter_root: str
    adapter_path: Optional[str]
    adapter_source: str
    available_checkpoints: list[str]
    using_adapter: bool
    device: str


def _resolve_project_root() -> Path:
    project_root = os.getenv("DEVGEN_PROJECT_ROOT")
    if project_root:
        return Path(project_root).expanduser().resolve()

    current_file = Path(__file__).resolve()
    if current_file.parent.name == "backend":
        return current_file.parent.parent
    return current_file.parent


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _has_adapter_files(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)", path.name)
    checkpoint_number = int(match.group(1)) if match else -1
    return checkpoint_number, path.name


def _read_adapter_base_model(adapter_path: Path) -> Optional[str]:
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        return None
    try:
        import json
        with config_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        value = data.get("base_model_name_or_path")
        return str(value) if value else None
    except Exception as exc:
        print(f"[TrOCR Engine] Could not read base model from {config_file}: {exc}")
        return None


def inspect_model_artifacts(
    base_model_name: Optional[str] = None,
    adapter_root: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    resolved_device = device or os.getenv("TROCR_DEVICE") or get_best_torch_device()
    explicit_base_model = base_model_name or os.getenv("TROCR_BASE_MODEL")
    project_root = _resolve_project_root()
    resolved_adapter_root = adapter_root or os.getenv("TROCR_ADAPTER_ROOT", DEFAULT_ADAPTER_ROOT)
    adapter_root_path = _resolve_path(resolved_adapter_root, project_root)

    checkpoints: list[Path] = []
    if adapter_root_path.exists():
        checkpoints = sorted(
            [path for path in adapter_root_path.glob("checkpoint-*") if _has_adapter_files(path)],
            key=_checkpoint_sort_key,
            reverse=True,
        )

    resolved_adapter_path: Optional[Path] = None
    adapter_source = "base_model_only"

    explicit_adapter = adapter_path or os.getenv("TROCR_ADAPTER_PATH")
    if explicit_adapter:
        explicit_path = _resolve_path(explicit_adapter, project_root)
        if _has_adapter_files(explicit_path):
            resolved_adapter_path = explicit_path
            adapter_source = "explicit_path"

    if resolved_adapter_path is None and _has_adapter_files(adapter_root_path):
        resolved_adapter_path = adapter_root_path
        adapter_source = "adapter_root"

    if resolved_adapter_path is None and checkpoints:
        resolved_adapter_path = checkpoints[0]
        adapter_source = "latest_checkpoint"

    adapter_base_model: Optional[str] = None
    if resolved_adapter_path is not None:
        adapter_base_model = _read_adapter_base_model(resolved_adapter_path)

    resolved_base_model = explicit_base_model or adapter_base_model or DEFAULT_BASE_MODEL

    artifacts = ModelArtifacts(
        project_root=str(project_root),
        base_model_name=resolved_base_model,
        adapter_root=str(adapter_root_path),
        adapter_path=str(resolved_adapter_path) if resolved_adapter_path else None,
        adapter_source=adapter_source,
        available_checkpoints=[checkpoint.name for checkpoint in checkpoints],
        using_adapter=resolved_adapter_path is not None,
        device=resolved_device,
    )
    return asdict(artifacts)


class TrOCREngine:
    """Segmentation-free OCR engine backed by TrOCR and an optional LoRA adapter.
    Also integrates a CNN character classifier for single-character inputs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_root: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        default_max_length: int = 128,
        enable_cnn: bool = True,
    ):
        self.device = device or os.getenv("TROCR_DEVICE") or get_best_torch_device()
        self.default_max_length = int(os.getenv("TROCR_MAX_LENGTH", str(default_max_length)))
        self.artifacts = inspect_model_artifacts(
            base_model_name=model_name,
            adapter_root=adapter_root,
            adapter_path=adapter_path,
            device=self.device,
        )
        self.base_model_name = self.artifacts["base_model_name"]
        self.adapter_path = self.artifacts["adapter_path"]

        print(
            f"[TrOCR Engine] Loading base model '{self.base_model_name}' on {self.device} "
            f"(adapter: {self.adapter_path or 'none'})..."
        )

        adapter_processor_source = None
        if self.adapter_path and (Path(self.adapter_path) / "preprocessor_config.json").exists():
            adapter_processor_source = self.adapter_path

        self.processor = load_trocr_processor(
            self.base_model_name,
            image_processor_source=adapter_processor_source,
        )
        base_model = VisionEncoderDecoderModel.from_pretrained(self.base_model_name)
        base_model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        base_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        base_model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        base_model.config.vocab_size = base_model.config.decoder.vocab_size

        if self.adapter_path:
            peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)
            try:
                self.model = peft_model.merge_and_unload()
            except Exception:
                self.model = peft_model
        else:
            self.model = base_model

        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            if self.device == "mps":
                print(f"[TrOCR Engine] Could not move model to MPS, falling back to CPU: {exc}")
                self.device = "cpu"
                self.artifacts["device"] = "cpu"
                self.model.to(self.device)
            else:
                raise
        self.model.eval()
        print("[TrOCR Engine] Model loaded successfully.")

        # Load CNN character classifier (optional, for single-char inputs)
        self.cnn_classifier = None
        if enable_cnn:
            try:
                from backend.cnn_model import CharacterClassifier
                self.cnn_classifier = CharacterClassifier(device=self.device)
            except Exception as exc:
                print(f"[TrOCR Engine] CNN classifier not available: {exc}")

    def info(self) -> dict:
        return {
            **self.artifacts,
            "loaded": True,
            "default_max_length": self.default_max_length,
            "num_beams": 4,
            "cnn_available": self.cnn_classifier is not None and self.cnn_classifier.available,
        }

    def recognize(self, image: Image.Image, max_length: Optional[int] = None,
                  force_model: Optional[str] = None) -> dict:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if force_model == "cnn" and self.cnn_classifier and self.cnn_classifier.available:
            use_cnn = True
            routing_info = {"type": "character", "confidence": 1.0, "reason": "forced"}
        elif force_model == "trocr":
            use_cnn = False
            routing_info = {"type": "word", "confidence": 1.0, "reason": "forced"}
        else:
            # Auto-routing
            from backend.image_router import classify_input_type
            routing_info = classify_input_type(image)
            if routing_info["type"] == "character" and self.cnn_classifier and self.cnn_classifier.available:
                use_cnn = True
            else:
                use_cnn = False

        # Route to CNN for single characters
        if use_cnn and self.cnn_classifier and self.cnn_classifier.available:
            result = self.cnn_classifier.predict(image)
            result["routing"] = routing_info
            result["tokens"] = [result["text"]]
            result["confidence_scores"] = [result["confidence"]]
            result["average_confidence"] = result["confidence"]
            result["generation_steps"] = 1
            result["model_info"] = {
                "base_model_name": self.base_model_name,
                "adapter_path": self.adapter_path,
                "using_adapter": bool(self.adapter_path),
                "device": self.device,
                "model_used": "cnn_classifier",
            }
            return result

        # Default: use TrOCR for words
        requested_max_length = max_length or self.default_max_length
        started_at = time.perf_counter()

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values,
                max_length=requested_max_length,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        token_data = self._decode_token_confidences(outputs)
        inference_ms = round((time.perf_counter() - started_at) * 1000, 2)

        return {
            "text": text,
            "tokens": token_data["tokens"],
            "confidence_scores": token_data["confidence_scores"],
            "average_confidence": token_data["average_confidence"],
            "inference_ms": inference_ms,
            "generation_steps": len(token_data["confidence_scores"]),
            "routing": routing_info,
            "model_info": {
                "base_model_name": self.base_model_name,
                "adapter_path": self.adapter_path,
                "using_adapter": bool(self.adapter_path),
                "device": self.device,
                "model_used": "trocr_lora",
            },
        }

    def _decode_token_confidences(self, outputs) -> dict:
        if not getattr(outputs, "scores", None):
            return {"tokens": [], "confidence_scores": [], "average_confidence": None}

        beam_indices = getattr(outputs, "beam_indices", None)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )

        special_token_ids = {
            getattr(self.processor.tokenizer, token_name)
            for token_name in SPECIAL_TOKEN_NAMES
            if getattr(self.processor.tokenizer, token_name, None) is not None
        }

        tokens: list[str] = []
        confidence_scores: list[float] = []
        generated_token_ids = outputs.sequences[0][1:]

        for token_id, log_score in zip(generated_token_ids, transition_scores[0]):
            token_value = token_id.item()
            if token_value in special_token_ids:
                continue

            decoded_token = self.processor.decode([token_value], skip_special_tokens=False).strip()
            if not decoded_token:
                continue

            confidence = round(float(torch.exp(log_score).item()), 4)
            tokens.append(decoded_token)
            confidence_scores.append(confidence)

        average_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores
            else None
        )
        return {
            "tokens": tokens,
            "confidence_scores": confidence_scores,
            "average_confidence": average_confidence,
        }

    def batch_recognize(self, images: list[Image.Image], max_length: Optional[int] = None) -> list[dict]:
        return [self.recognize(image, max_length=max_length) for image in images]


class CERCalculator:
    """Character Error Rate calculator using Levenshtein Distance."""

    @staticmethod
    def calculate(prediction: str, reference: str) -> dict:
        import editdistance

        pred_chars = list(prediction)
        ref_chars = list(reference)

        edit_dist = editdistance.eval(pred_chars, ref_chars)
        reference_length = len(ref_chars)
        cer = edit_dist / reference_length if reference_length > 0 else 0.0

        return {
            "cer": round(cer, 4),
            "edit_distance": edit_dist,
            "reference_length": reference_length,
            "prediction_length": len(pred_chars),
        }
