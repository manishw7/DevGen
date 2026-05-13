"""
DevGen Framework — CNN Devanagari Character Classifier

A lightweight CNN for classifying individual handwritten Devanagari
characters (vowels, consonants, digits) — 46 classes total.

This model complements TrOCR (which handles words) by handling single
characters that TrOCR hallucinates on.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── 46-class label map ──────────────────────────────────────────────────────
# Standard DHCD ordering: 36 consonants/vowels + 10 digits
DEVANAGARI_CLASSES = [
    # Consonants (ka to gya)
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "क्ष", "त्र", "ज्ञ",
    # Digits (0-9)
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
]

# Reverse map: character → index
CHAR_TO_INDEX = {ch: i for i, ch in enumerate(DEVANAGARI_CLASSES)}
NUM_CLASSES = len(DEVANAGARI_CLASSES)

# Default model path
DEFAULT_CNN_MODEL_PATH = "devanagari-cnn-classifier.pt"


class DevanagariCNN(nn.Module):
    """
    3-layer CNN for 32×32 grayscale character images.
    ~500K parameters — fast inference even on CPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Inference transform ─────────────────────────────────────────────────────
# Matches training: resize to 32×32, grayscale, normalize
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class CharacterClassifier:
    """
    Wrapper for loading and running the trained CNN model.
    Used by the smart router in TrOCREngine.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # Find model file
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = str(project_root / DEFAULT_CNN_MODEL_PATH)

        self.model_path = model_path
        self.model: Optional[DevanagariCNN] = None
        self.available = False

        if os.path.exists(model_path):
            self._load_model()
        else:
            print(f"[CNN Classifier] Model not found at {model_path} — single character recognition disabled")

    def _load_model(self):
        """Load the trained CNN weights."""
        try:
            self.model = DevanagariCNN(NUM_CLASSES)
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            size_mb = os.path.getsize(self.model_path) / 1e6
            print(f"[CNN Classifier] Loaded ({size_mb:.1f} MB) on {self.device} — {NUM_CLASSES} classes")
        except Exception as exc:
            print(f"[CNN Classifier] Failed to load model: {exc}")
            self.model = None
            self.available = False

    def predict(self, image: Image.Image) -> dict:
        """
        Classify a single character image.

        Returns:
            dict with text, confidence, class_index, model_used
        """
        if not self.available or self.model is None:
            return {"text": "", "confidence": 0.0, "error": "CNN model not loaded"}

        started_at = time.perf_counter()

        # Preprocess using DHCD style
        tensor = self._preprocess_dhcd_style(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        predicted_char = DEVANAGARI_CLASSES[pred_idx.item()]
        conf_value = round(confidence.item(), 4)
        inference_ms = round((time.perf_counter() - started_at) * 1000, 2)

        # Top-3 predictions for debugging
        top3_probs, top3_indices = probs.topk(3, dim=1)
        top3 = [
            {"char": DEVANAGARI_CLASSES[idx.item()], "confidence": round(prob.item(), 4)}
            for idx, prob in zip(top3_indices[0], top3_probs[0])
        ]

        return {
            "text": predicted_char,
            "confidence": conf_value,
            "class_index": pred_idx.item(),
            "top3": top3,
            "inference_ms": inference_ms,
            "model_used": "cnn_classifier",
        }

    def _preprocess_dhcd_style(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses a character image to match DHCD dataset (inverted, tightly cropped, padded)."""
        import cv2
        import numpy as np
        
        # Convert PIL to CV2 grayscale
        img = np.array(image.convert("L"))
        
        # Binarize and invert (DHCD is white ink on black background)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Crop to bounding box
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = binary[y:y+h, x:x+w]
        else:
            cropped = binary
            h, w = cropped.shape
            
        # Pad to square and add 16px border (helps CNN focus on center)
        side = max(w, h)
        padded = np.zeros((side + 16, side + 16), dtype=np.uint8)
        y_off = (side + 16 - h) // 2
        x_off = (side + 16 - w) // 2
        padded[y_off:y_off+h, x_off:x_off+w] = cropped
        
        # Resize to 32x32
        resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor
