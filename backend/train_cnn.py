"""
DevGen Framework — CNN Character Classifier Training Script

Trains DevanagariCNN on the DHCD dataset (92K images, 46 classes).
Supports both CSV format (from Kaggle) and ImageFolder format.
Runs locally on MPS/CUDA/CPU — takes ~5-10 minutes.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m backend.train_cnn
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.cnn_model import DevanagariCNN, DEVANAGARI_CLASSES, NUM_CLASSES

# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_CSV_PATH = "data/data.csv"
OUTPUT_MODEL = "devanagari-cnn-classifier.pt"
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
IMAGE_SIZE = 32
TRAIN_SPLIT = 0.85

# DHCD folder-name → Devanagari character mapping
DHCD_FOLDER_MAP = {
    "character_01_ka": "क", "character_02_kha": "ख", "character_03_ga": "ग",
    "character_04_gha": "घ", "character_05_kna": "ङ",
    "character_06_cha": "च", "character_07_chha": "छ", "character_08_ja": "ज",
    "character_09_jha": "झ", "character_10_yna": "ञ",
    "character_11_taamatar": "ट", "character_12_thaa": "ठ",
    "character_13_daa": "ड", "character_14_dhaa": "ढ", "character_15_adna": "ण",
    "character_16_tabala": "त", "character_17_tha": "थ",
    "character_18_da": "द", "character_19_dha": "ध", "character_20_na": "न",
    "character_21_pa": "प", "character_22_pha": "फ",
    "character_23_ba": "ब", "character_24_bha": "भ", "character_25_ma": "म",
    "character_26_yaw": "य", "character_27_ra": "र",
    "character_28_la": "ल", "character_29_waw": "व",
    "character_30_motosaw": "श", "character_31_petchiryakha": "ष",
    "character_32_patalosaw": "स", "character_33_ha": "ह",
    "character_34_chhya": "क्ष", "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",
    "digit_0": "०", "digit_1": "१", "digit_2": "२", "digit_3": "३",
    "digit_4": "४", "digit_5": "५", "digit_6": "६", "digit_7": "७",
    "digit_8": "८", "digit_9": "९",
}


class DHCDDataset(Dataset):
    """Load DHCD from CSV: each row = 1024 pixel values + class label."""

    def __init__(self, csv_path: str, transform=None):
        import csv

        self.transform = transform
        self.images = []
        self.labels = []

        # Build label mapping: folder_name → index (matching DEVANAGARI_CLASSES)
        char_to_idx = {ch: i for i, ch in enumerate(DEVANAGARI_CLASSES)}
        folder_to_idx = {}
        for folder, char in DHCD_FOLDER_MAP.items():
            if char in char_to_idx:
                folder_to_idx[folder] = char_to_idx[char]

        skipped = 0
        print(f"Loading CSV from {csv_path}...")

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                label_name = row[-1].strip()
                if label_name not in folder_to_idx:
                    skipped += 1
                    continue

                pixels = np.array([int(x) for x in row[:-1]], dtype=np.uint8)
                img = pixels.reshape(32, 32)
                self.images.append(img)
                self.labels.append(folder_to_idx[label_name])

        print(f"Loaded {len(self.images)} samples ({skipped} skipped)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        from PIL import Image
        pil_img = Image.fromarray(img, mode="L")

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        return tensor, self.labels[idx]


# ── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
            return "mps"
        except RuntimeError:
            pass
    return "cpu"


def train(csv_path: str, output_path: str, epochs: int = EPOCHS):
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Classes: {NUM_CLASSES}")

    # Load full dataset from CSV
    full_dataset = DHCDDataset(csv_path, transform=None)

    # Split into train/test
    n_train = int(len(full_dataset) * TRAIN_SPLIT)
    n_test = len(full_dataset) - n_train

    train_indices, test_indices = random_split(
        range(len(full_dataset)), [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Create datasets with different transforms
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = list(indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img = self.dataset.images[real_idx]
            label = self.dataset.labels[real_idx]
            from PIL import Image
            pil_img = Image.fromarray(img, mode="L")
            tensor = self.transform(pil_img)
            return tensor, label

    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    test_dataset = TransformSubset(full_dataset, test_indices, test_transform)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device != "cpu"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device != "cpu"),
    )

    # Model
    model = DevanagariCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\n{'='*60}")
    print(f"  TRAINING")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = train_correct / train_total
        avg_loss = train_loss / train_total

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = test_correct / test_total
        elapsed = time.time() - epoch_start

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc*100:.1f}% | "
            f"Test: {test_acc*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_path)
            print(f"    → Saved best model ({test_acc*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best test accuracy: {best_acc*100:.1f}%")
    print(f"  Model saved to: {output_path}")
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Model size: {size_mb:.1f} MB")
    print(f"{'='*60}")

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Devanagari CNN character classifier")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to DHCD data.csv")
    parser.add_argument("--output", default=OUTPUT_MODEL, help="Output model path")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    args = parser.parse_args()

    train(args.csv, args.output, args.epochs)
