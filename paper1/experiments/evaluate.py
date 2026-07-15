"""
Evaluate a TrOCR checkpoint (base / base+LoRA / full fine-tune) on the
official IIIT-INDIC-HW-WORDS-Hindi test split.

Writes a results JSON with aggregate metrics (CER + 95% bootstrap CI, akshara
error rate, word accuracy) and per-sample predictions for error analysis.

Examples:
    # Unadapted base checkpoint
    python -m paper1.experiments.evaluate --run-name base_checkpoint

    # Shipped LoRA adapter
    python -m paper1.experiments.evaluate \
        --adapter-path trocr-devanagari-lora-hf --run-name lora_r16_legacy

    # Full fine-tune checkpoint
    python -m paper1.experiments.evaluate \
        --full-model-path ./trocr-devanagari-full --run-name full_ft
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper1.experiments.common import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_BASE_MODEL,
    RESULTS_DIR,
    EvalSummary,
    akshara_error_rate,
    bootstrap_ci,
    cer,
    count_params,
    load_model,
    normalize_text,
    save_results,
    set_seed,
    word_error,
)


def coerce_to_pil(image_data) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_data)!r}")


def load_training_config(*paths: str | None) -> dict:
    """Load the run_config.json saved by train_trocr.py, if this is a trained run."""
    for raw_path in paths:
        if not raw_path:
            continue
        config_path = Path(raw_path) / "run_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR on the official test split.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter directory.")
    parser.add_argument("--full-model-path", default=None, help="Fully fine-tuned model directory.")
    parser.add_argument("--parquet-dir", default=None,
                        help="Optional local HF parquet directory containing <split>-*.parquet files.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Generation max length. Defaults to the checkpoint generation config.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate a subset (smoke test).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True, help="Identifier used in the results filename.")
    parser.add_argument("--app-preprocess", action="store_true",
                        help="Apply the serving pipeline's crop/pad preprocessing before the "
                             "processor (measures deployment-pipeline effect; off = raw eval).")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, processor, device = load_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        full_model_path=args.full_model_path,
        device=args.device,
    )
    trainable, total = count_params(model)
    training_config = load_training_config(args.adapter_path, args.full_model_path)
    if training_config:
        trainable = training_config.get("trainable_params", trainable)
        total = training_config.get("total_params", total)
    print(f"[evaluate] Model on {device} — trainable {trainable:,} / total {total:,}")

    split_spec = f"train[:{args.limit}]" if args.limit else "train"
    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        files = sorted(str(path) for path in parquet_dir.glob(f"{args.split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found for {args.split!r} in {parquet_dir}")
        dataset = load_dataset("parquet", data_files=files, split=split_spec)
    else:
        remote_split_spec = f"{args.split}[:{args.limit}]" if args.limit else args.split
        dataset = load_dataset(DATASET_NAME, split=remote_split_spec)
    print(f"[evaluate] {len(dataset)} samples from split '{args.split}'")
    max_length = args.max_length or getattr(model.generation_config, "max_length", None) or 128

    app_preprocess = None
    if args.app_preprocess:
        from backend.preprocessing import preprocess_pil_for_ocr as app_preprocess

    samples: list[dict] = []
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        images = [coerce_to_pil(img) for img in batch["image"]]
        if app_preprocess:
            images = [app_preprocess(img) for img in images]
        references = [str(t) for t in batch["text"]]

        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=pixel_values,
                max_length=max_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
        predictions = processor.batch_decode(output_ids, skip_special_tokens=True)

        for offset, (pred, ref) in enumerate(zip(predictions, references)):
            samples.append({
                "index": start + offset,
                "reference": normalize_text(ref),
                "prediction": normalize_text(pred),
                "cer": round(cer(pred, ref), 5),
                "aer": round(akshara_error_rate(pred, ref), 5),
                "word_error": word_error(pred, ref),
            })
        done = min(start + args.batch_size, len(dataset))
        if done % (args.batch_size * 20) < args.batch_size or done == len(dataset):
            running_cer = sum(s["cer"] for s in samples) / len(samples)
            print(f"[evaluate] {done}/{len(dataset)}  running CER={running_cer:.4f}")

    cers = [s["cer"] for s in samples]
    summary = EvalSummary(
        run_name=args.run_name,
        num_samples=len(samples),
        cer=sum(cers) / len(cers),
        cer_ci95=bootstrap_ci(cers, seed=args.seed),
        aer=sum(s["aer"] for s in samples) / len(samples),
        word_accuracy=1.0 - sum(s["word_error"] for s in samples) / len(samples),
        trainable_params=trainable,
        total_params=total,
        config={
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "full_model_path": args.full_model_path,
            "split": args.split,
            "num_beams": args.num_beams,
            "max_length": max_length,
            "app_preprocess": args.app_preprocess,
            "seed": args.seed,
            "device": device,
            "limit": args.limit,
            "training_config": training_config,
        },
    )

    print(f"\n[evaluate] {args.run_name}")
    print(f"  CER            {summary.cer:.4f}  (95% CI {summary.cer_ci95[0]:.4f}–{summary.cer_ci95[1]:.4f})")
    print(f"  Akshara ER     {summary.aer:.4f}")
    print(f"  Word accuracy  {summary.word_accuracy:.4f}")

    output_path = Path(args.output_dir) / f"{args.run_name}.json"
    save_results({"summary": summary.to_dict(), "samples": samples}, output_path)


if __name__ == "__main__":
    main()
