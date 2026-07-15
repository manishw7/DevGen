"""
Generate a synthetic Devanagari handwriting training set with the trained
ControlNet LDM, in the extracted-dataset format train_trocr.py consumes
(images/ + labels.csv).

Vocabulary is sampled from the real train split's word list so synthetic data
matches the target distribution; use --vocab-file for OOV augmentation.
Each image gets a distinct seed -> distinct handwriting instance per word.

Usage:
    python -m paper1.experiments.generate_synthetic \
        --count 5000 --out-dir data_synth/ldm_5k

Requires the ControlNet checkpoint under ldm/ (see backend/ldm_engine.py)
and a GPU/MPS machine — ~2-4s per image at 25 steps.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper1.experiments.common import DATASET_NAME, normalize_text, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic LDM training data.")
    parser.add_argument("--count", type=int, default=5000, help="Number of images to generate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-file", default=None,
                        help="Optional newline-separated word list; default samples train-split words.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--save-control", action="store_true",
                        help="Also save the rendered-glyph control images (for figures).")
    parser.add_argument("--one-per-word", action="store_true",
                        help="Generate exactly one image per vocabulary word in order "
                             "(evaluation sets); --count is ignored.")
    return parser.parse_args()


def load_vocabulary(vocab_file: str | None) -> list[str]:
    if vocab_file:
        words = [normalize_text(line) for line in Path(vocab_file).read_text(encoding="utf-8").splitlines()]
        return [w for w in words if w]
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="train")
    return [normalize_text(t) for t in dataset["text"] if str(t).strip()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from backend.ldm_engine import generate_handwriting_ldm

    vocabulary = load_vocabulary(args.vocab_file)
    print(f"[generate_synthetic] Vocabulary size: {len(vocabulary)} words")

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    control_dir = out_dir / "control"
    if args.save_control:
        control_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    total = len(vocabulary) if args.one_per_word else args.count
    for i in range(total):
        word = vocabulary[i] if args.one_per_word else rng.choice(vocabulary)
        image_seed = args.seed * 1_000_003 + i  # deterministic, unique per image
        output_image, control_image = generate_handwriting_ldm(
            word,
            seed=image_seed,
            num_inference_steps=args.steps,
            conditioning_scale=args.conditioning_scale,
        )
        filename = f"synth_{i:06d}.png"
        output_image.save(images_dir / filename)
        if args.save_control:
            control_image.save(control_dir / filename)
        rows.append({"filename": filename, "text": word, "seed": str(image_seed)})
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"[generate_synthetic] {i + 1}/{total}")

    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "text", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[generate_synthetic] Wrote {len(rows)} samples to {out_dir}")
    print(f"[generate_synthetic] Use with: python backend/train_trocr.py --synthetic-dir {out_dir}")


if __name__ == "__main__":
    main()
