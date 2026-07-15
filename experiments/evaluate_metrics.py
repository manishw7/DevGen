"""
DevGen — Diagnostic Metric Evaluation (Report Tables 5.5 / 5.6)

Computes:
  TrOCR + LoRA (Table 5.5):
    - Character Error Rate (CER)
    - Word Exact Match (%)
    - Character Precision / Recall / F1 (multiset intersection, micro-averaged)
  Custom CNN (Table 5.6):
    - Accuracy on DHCD split
    - Macro Precision / Recall / F1

Model paths are provided separately via --trocr-adapter and --cnn-model.
The IIIT-INDIC-HW-WORDS-Hindi split (validation or test) is downloaded from
Hugging Face (or read from a local parquet dir). The DHCD dataset is
downloaded from the UCI repository if not already present.

Examples
--------
# Both models, 50 random test samples for TrOCR, 2000 random DHCD test samples
python experiments/evaluate_metrics.py \
    --trocr-adapter trocr-devanagari-lora-hf/checkpoint-14000 \
    --cnn-model devanagari-cnn-classifier.pt \
    --split test --num-samples 50 \
    --dhcd-split Test --dhcd-samples 2000

# TrOCR only, full validation set, local parquet files
python experiments/evaluate_metrics.py \
    --trocr-adapter trocr-devanagari-lora-hf/checkpoint-14000 \
    --data-dir data/iiit_hindi_parquet --split validation --num-samples 0

# CNN only, full DHCD Test split
python experiments/evaluate_metrics.py \
    --cnn-model devanagari-cnn-classifier.pt --dhcd-samples 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.cnn_model import DEVANAGARI_CLASSES, NUM_CLASSES, DevanagariCNN  # noqa: E402

DATASET_NAME = "c3rl/IIIT-INDIC-HW-WORDS-Hindi"
DHCD_URL = (
    "https://archive.ics.uci.edu/static/public/389/"
    "devanagari+handwritten+character+dataset.zip"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ────────────────────────── metric primitives ──────────────────────────────

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def char_prf(pred: str, ref: str) -> tuple[int, int, int]:
    """Multiset character intersection counts: (matched, |pred|, |ref|)."""
    pc, rc = Counter(pred), Counter(ref)
    matched = sum((pc & rc).values())
    return matched, len(pred), len(ref)


# ────────────────────────── TrOCR evaluation ───────────────────────────────

def load_trocr(adapter_path: str, base_model: str, device: str):
    from peft import PeftModel
    from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

    adapter = Path(adapter_path).expanduser().resolve()
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter}")

    # Prefer the base model recorded inside the adapter config
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    base = cfg.get("base_model_name_or_path") or base_model
    print(f"[TrOCR] base={base}  adapter={adapter}")

    model = VisionEncoderDecoderModel.from_pretrained(base)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.to(device).eval()

    try:
        processor = TrOCRProcessor.from_pretrained(base)
    except Exception:
        image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        tokenizer = AutoTokenizer.from_pretrained(base)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return model, processor


def load_word_dataset(split: str, data_dir: str | None, num_samples: int, seed: int):
    from datasets import load_dataset

    if data_dir:
        files = sorted(str(p) for p in Path(data_dir).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split}-*.parquet under {data_dir}")
        ds = load_dataset("parquet", data_files=files, split="train")
    else:
        ds = load_dataset(DATASET_NAME, split=split)

    if num_samples and num_samples < len(ds):
        indices = random.Random(seed).sample(range(len(ds)), num_samples)
        ds = ds.select(indices)
    return ds


@torch.inference_mode()
def evaluate_trocr(args, device: str) -> dict:
    model, processor = load_trocr(args.trocr_adapter, args.trocr_base, device)
    ds = load_word_dataset(args.split, args.data_dir, args.num_samples, args.seed)
    print(f"[TrOCR] Evaluating {len(ds)} samples from split '{args.split}'")

    total_edits = total_ref_chars = 0
    exact = 0
    matched_sum = pred_len_sum = ref_len_sum = 0
    rows = []

    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        refs = [str(t).strip() for t in batch["text"]]

        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        generated = model.generate(pixel_values, num_beams=4, max_length=64)
        preds = [p.strip() for p in processor.batch_decode(generated, skip_special_tokens=True)]

        for pred, ref in zip(preds, refs):
            edits = levenshtein(pred, ref)
            total_edits += edits
            total_ref_chars += len(ref)
            exact += int(pred == ref)
            m, pl, rl = char_prf(pred, ref)
            matched_sum += m
            pred_len_sum += pl
            ref_len_sum += rl
            rows.append({"reference": ref, "prediction": pred, "cer": round(edits / max(len(ref), 1), 4)})

        done = min(start + args.batch_size, len(ds))
        print(f"  {done}/{len(ds)}  running CER={total_edits / max(total_ref_chars, 1):.4f}")

    precision = matched_sum / max(pred_len_sum, 1)
    recall = matched_sum / max(ref_len_sum, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "samples": len(ds),
        "character_error_rate": round(total_edits / max(total_ref_chars, 1), 4),
        "word_exact_match_pct": round(100.0 * exact / len(ds), 2),
        "character_precision": round(precision, 4),
        "character_recall": round(recall, 4),
        "character_f1": round(f1, 4),
        "predictions": rows,
    }


# ────────────────────────── DHCD / CNN evaluation ──────────────────────────

def ensure_dhcd(dhcd_dir: Path) -> Path:
    """Download + extract DHCD from UCI if not already present. Returns dataset root."""
    for candidate in (dhcd_dir / "DevanagariHandwrittenCharacterDataset", dhcd_dir):
        if (candidate / "Train").is_dir() and (candidate / "Test").is_dir():
            return candidate

    dhcd_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dhcd_dir / "dhcd.zip"
    if not zip_path.exists():
        print(f"[DHCD] Downloading {DHCD_URL} ...")
        urllib.request.urlretrieve(DHCD_URL, zip_path)
    print("[DHCD] Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dhcd_dir)

    for candidate in (dhcd_dir / "DevanagariHandwrittenCharacterDataset", dhcd_dir):
        if (candidate / "Train").is_dir():
            return candidate
    raise FileNotFoundError(f"Train/Test folders not found under {dhcd_dir} after extraction")


def dhcd_folder_to_index(folder_name: str) -> int | None:
    """Map DHCD class folder → index in DEVANAGARI_CLASSES (ka..gya=0-35, digits=36-45)."""
    name = folder_name.lower()
    if name.startswith("character_"):
        return int(name.split("_")[1]) - 1
    if name.startswith("digit_"):
        return 36 + int(name.split("_")[1])
    return None


def collect_dhcd_samples(root: Path, split: str, num_samples: int, seed: int) -> list[tuple[Path, int]]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        idx = dhcd_folder_to_index(class_dir.name)
        if idx is None or idx >= NUM_CLASSES:
            print(f"[DHCD] Skipping unrecognized folder: {class_dir.name}")
            continue
        samples.extend((p, idx) for p in class_dir.glob("*.png"))

    if num_samples and num_samples < len(samples):
        samples = random.Random(seed).sample(samples, num_samples)
    return samples


@torch.inference_mode()
def evaluate_cnn(args, device: str) -> dict:
    model = DevanagariCNN(NUM_CLASSES)
    state_dict = torch.load(args.cnn_model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[CNN] Loaded {args.cnn_model} on {device}")

    root = ensure_dhcd(Path(args.dhcd_dir))
    samples = collect_dhcd_samples(root, args.dhcd_split, args.dhcd_samples, args.seed)
    print(f"[CNN] Evaluating {len(samples)} DHCD '{args.dhcd_split}' images")

    # DHCD images are already white-ink-on-black 32×32 — match training transform
    def to_tensor(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L").resize((32, 32))
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().view(1, 32, 32)
        return (t / 255.0 - 0.5) / 0.5

    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    correct = 0

    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        batch = torch.stack([to_tensor(p) for p, _ in chunk]).to(device)
        labels = torch.tensor([y for _, y in chunk])
        preds = model(batch).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        for t, p in zip(labels, preds):
            confusion[t, p] += 1

    # Macro precision / recall / F1 over classes that appear in the sample
    precisions, recalls, f1s = [], [], []
    for c in range(NUM_CLASSES):
        support = confusion[c].sum().item()
        if support == 0:
            continue
        tp = confusion[c, c].item()
        predicted = confusion[:, c].sum().item()
        p = tp / predicted if predicted else 0.0
        r = tp / support
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)

    per_class_acc = {
        DEVANAGARI_CLASSES[c]: round(confusion[c, c].item() / confusion[c].sum().item(), 4)
        for c in range(NUM_CLASSES)
        if confusion[c].sum().item() > 0
    }

    n = len(precisions)
    return {
        "samples": len(samples),
        "accuracy_pct": round(100.0 * correct / len(samples), 2),
        "macro_precision": round(sum(precisions) / n, 4),
        "macro_recall": round(sum(recalls) / n, 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "per_class_accuracy": per_class_acc,
    }


# ────────────────────────────── main ────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DevGen diagnostic metric evaluation (Tables 5.5 / 5.6)")
    # TrOCR
    parser.add_argument("--trocr-adapter", help="Path to LoRA adapter dir (contains adapter_config.json)")
    parser.add_argument("--trocr-base", default="paudelanil/trocr-devanagari-2", help="Base TrOCR model")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="IIIT-HW-Hindi split")
    parser.add_argument("--data-dir", help="Local parquet dir (e.g. data/iiit_hindi_parquet); omit to download from HF")
    parser.add_argument("--num-samples", type=int, default=50, help="Random word samples (0 = full split)")
    # CNN
    parser.add_argument("--cnn-model", help="Path to CNN state_dict (.pt)")
    parser.add_argument("--dhcd-dir", default="data/dhcd", help="DHCD root; auto-downloaded from UCI if missing")
    parser.add_argument("--dhcd-split", choices=["Train", "Test"], default="Test", help="DHCD split")
    parser.add_argument("--dhcd-samples", type=int, default=2000, help="Random DHCD samples (0 = full split)")
    # Common
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/metrics_results.json", help="JSON results path")
    args = parser.parse_args()

    if not args.trocr_adapter and not args.cnn_model:
        parser.error("Provide --trocr-adapter and/or --cnn-model")

    device = get_device()
    print(f"[Eval] device={device}  seed={args.seed}")
    results: dict = {"device": device, "seed": args.seed}

    if args.trocr_adapter:
        t0 = time.perf_counter()
        results["trocr"] = evaluate_trocr(args, device)
        results["trocr"]["eval_seconds"] = round(time.perf_counter() - t0, 1)

    if args.cnn_model:
        t0 = time.perf_counter()
        results["cnn"] = evaluate_cnn(args, device)
        results["cnn"]["eval_seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    if "trocr" in results:
        r = results["trocr"]
        print(f"TrOCR + LoRA  ({r['samples']} samples, split={args.split})")
        print(f"  Character Error Rate : {r['character_error_rate']}")
        print(f"  Word Exact Match     : {r['word_exact_match_pct']}%")
        print(f"  Character Precision  : {r['character_precision']}")
        print(f"  Character Recall     : {r['character_recall']}")
        print(f"  Character F1-Score   : {r['character_f1']}")
    if "cnn" in results:
        r = results["cnn"]
        print(f"Custom CNN  ({r['samples']} samples, DHCD {args.dhcd_split})")
        print(f"  Accuracy             : {r['accuracy_pct']}%")
        print(f"  Macro Precision      : {r['macro_precision']}")
        print(f"  Macro Recall         : {r['macro_recall']}")
        print(f"  Macro F1-Score       : {r['macro_f1']}")
    print(f"\nFull results (incl. per-sample predictions) → {out}")


if __name__ == "__main__":
    main()
