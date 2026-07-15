"""
Content fidelity for Paper 2: read every generated image with a Devanagari
TrOCR recognizer (best Paper-1 adapter) and score CER/AER/word accuracy
against the intended text. High scores mean generations are readable as the
right word.

Input: a generated set in extracted format (images/ + labels.csv from
paper1/experiments/generate_synthetic.py).

Usage:
    python -m paper2.experiments.content_fidelity \
        --generated-dir data_synth/eval_iv \
        --adapter-path paper1/runs/lora_r16_attn \
        --run-name fidelity_iv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper1.experiments.common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    EvalSummary,
    akshara_error_rate,
    bootstrap_ci,
    cer,
    load_model,
    normalize_text,
    save_results,
    set_seed,
    word_error,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content fidelity of generated handwriting.")
    parser.add_argument("--generated-dir", required=True, help="Directory with images/ + labels.csv.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=None, help="Judge recognizer LoRA adapter.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--app-preprocess", action="store_true", default=True,
                        help="Apply the recognizer's crop/pad preprocessing (default on).")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, processor, device = load_model(
        base_model=args.base_model, adapter_path=args.adapter_path,
    )

    generated_dir = Path(args.generated_dir)
    with (generated_dir / "labels.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[content_fidelity] {len(rows)} generated images from {generated_dir}")

    preprocess = None
    if args.app_preprocess:
        from backend.preprocessing import preprocess_pil_for_ocr as preprocess

    samples: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = [Image.open(generated_dir / "images" / r["filename"]).convert("RGB") for r in batch]
        if preprocess:
            images = [preprocess(img) for img in images]
        references = [str(r["text"]) for r in batch]

        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
        predictions = processor.batch_decode(output_ids, skip_special_tokens=True)

        for row, pred, ref in zip(batch, predictions, references):
            samples.append({
                "filename": row["filename"],
                "reference": normalize_text(ref),
                "prediction": normalize_text(pred),
                "cer": round(cer(pred, ref), 5),
                "aer": round(akshara_error_rate(pred, ref), 5),
                "word_error": word_error(pred, ref),
            })

    cers = [s["cer"] for s in samples]
    summary = EvalSummary(
        run_name=args.run_name,
        num_samples=len(samples),
        cer=sum(cers) / len(cers),
        cer_ci95=bootstrap_ci(cers, seed=args.seed),
        aer=sum(s["aer"] for s in samples) / len(samples),
        word_accuracy=1.0 - sum(s["word_error"] for s in samples) / len(samples),
        config={
            "generated_dir": str(generated_dir),
            "judge_base_model": args.base_model,
            "judge_adapter": args.adapter_path,
            "num_beams": args.num_beams,
            "seed": args.seed,
        },
    )
    print(f"[content_fidelity] {args.run_name}: CER {summary.cer:.4f}  "
          f"AER {summary.aer:.4f}  WAcc {summary.word_accuracy:.4f}")
    save_results({"summary": summary.to_dict(), "samples": samples},
                 Path(args.output_dir) / f"{args.run_name}.json")


if __name__ == "__main__":
    main()
