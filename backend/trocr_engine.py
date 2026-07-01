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
import cv2
import numpy as np
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

DEFAULT_BASE_MODEL = "paudelanil/trocr-devanagari-2"
DEFAULT_ADAPTER_ROOT = "trocr-devanagari-lora-hf"
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
    # Try loading from the adapter source first if provided
    if image_processor_source:
        try:
            image_processor = ViTImageProcessor.from_pretrained(image_processor_source)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception:
            pass
            
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except Exception as exc:
        print(f"[TrOCR Engine] Processor fallback for '{model_name}': {exc}")
        # Manually configure the processor to match the HF space's stable settings
        image_processor = ViTImageProcessor.from_pretrained(DEFAULT_IMAGE_PROCESSOR)
        # Force specific normalization parity
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        
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
            # We run as a pure PeftModel instead of merging to prevent MPS numerical drift
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
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

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply HF-style preprocessing: cropping and aspect-ratio preserving padding."""
        # Convert PIL to OpenCV
        img = np.array(image.convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        aspect_ratio = w / float(h)

        # Helper: Crop to foreground
        def crop_to_foreground(img_cv, padding_ratio=0.18):
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return img_cv
            min_area = max(12, int(gray.shape[0] * gray.shape[1] * 0.0001))
            boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
            if not boxes: return img_cv
            x1, y1, x2, y2 = min(x for x,_,_,_ in boxes), min(y for _,y,_,_ in boxes), max(x+bw for x,_,bw,_ in boxes), max(y+bh for _,y,_,bh in boxes)
            pad_x, pad_y = max(8, int((x2 - x1) * padding_ratio)), max(8, int((y2 - y1) * padding_ratio))
            x1, y1, x2, y2 = max(0, x1-pad_x), max(0, y1-pad_y), min(gray.shape[1], x2+pad_x), min(gray.shape[0], y2+pad_y)
            return img_cv[y1:y2, x1:x2]

        # Helper: Normalize to square (padding)
        def normalize_to_square(img_cv, size=384):
            ih, iw = img_cv.shape[:2]
            scale = min(size / ih, size / iw)
            nh, nw = int(ih * scale), int(iw * scale)
            resized = cv2.resize(img_cv, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
            y_off, x_off = (size - nh) // 2, (size - nw) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
            return canvas

        # Apply logic based on aspect ratio
        if aspect_ratio <= 1.55:
            img = crop_to_foreground(img)
        elif aspect_ratio > 2.2:
            img = crop_to_foreground(img)
            img = normalize_to_square(img)
        
        # Convert back to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def recognize(self, image: Image.Image, max_length: Optional[int] = None,
                  force_model: Optional[str] = None) -> dict:
        """
        Recognize text from an image. 
        `force_model` can be: None (Auto), "cnn" (Character), or "trocr" (Word).
        """
        # Step 1: Preprocess (HF-style byte-level processing)
        import io
        from backend.preprocessing import preprocess_for_ocr
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        # This ensures parity with the HF space's preprocessing logic
        processed_image = preprocess_for_ocr(buf.getvalue())

        # Step 2: Routing (Manual or Auto)
        print(f"[TrOCR Engine] recognize: force_model={force_model}")
        if force_model == "cnn" and self.cnn_classifier and self.cnn_classifier.available:
            use_cnn = True
            routing_info = {"type": "character", "confidence": 1.0, "reason": "forced"}
        elif force_model == "trocr":
            use_cnn = False
            routing_info = {"type": "word", "confidence": 1.0, "reason": "forced"}
        else:
            # Auto-routing using the preprocessed image for better accuracy
            from backend.image_router import classify_input_type
            routing_info = classify_input_type(processed_image)
            if routing_info["type"] == "character" and self.cnn_classifier and self.cnn_classifier.available:
                use_cnn = True
            else:
                use_cnn = False
        print(f"[TrOCR Engine] Routing decision: use_cnn={use_cnn}, type={routing_info['type']}")

        # Route to CNN for single characters
        if use_cnn and self.cnn_classifier and self.cnn_classifier.available:
            result = self.cnn_classifier.predict(processed_image)
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
                "force_model_received": force_model,
            }
            return result

        # Default: use TrOCR for words
        requested_max_length = max_length or self.default_max_length
        started_at = time.perf_counter()

        pixel_values = self.processor(images=processed_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=pixel_values,
                max_length=requested_max_length,
                num_beams=4,
                early_stopping=True,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
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
                "force_model_received": force_model,
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
