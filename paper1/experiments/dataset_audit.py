"""
Audit the exact IIIT-INDIC-HW-WORDS-Hindi splits used by Paper 1.

By default this uses the Hugging Face dataset API, so it reports split sizes
without downloading the 1.9 GB image dataset. Add --include-label-stats when
you are on a GPU/cloud machine or already have the dataset cached locally.

Usage:
    python -m paper1.experiments.dataset_audit
    python -m paper1.experiments.dataset_audit --include-label-stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset

from paper1.experiments.akshara import split_aksharas
from paper1.experiments.common import DATASET_NAME, RESULTS_DIR, normalize_text, save_results

HF_API_URL = f"https://huggingface.co/api/datasets/{DATASET_NAME}"


def load_split_metadata() -> list[dict]:
    response = requests.get(HF_API_URL, timeout=30)
    response.raise_for_status()
    card_data = response.json().get("cardData") or {}
    dataset_info = card_data.get("dataset_info") or {}
    return [
        {
            "split": split["name"],
            "num_samples": split["num_examples"],
            "num_bytes": split.get("num_bytes"),
        }
        for split in dataset_info.get("splits", [])
    ]


def summarize_labels(split: str) -> dict:
    dataset = load_dataset(DATASET_NAME, split=split)
    labels = [normalize_text(text) for text in dataset["text"]]
    char_lengths = np.asarray([len(label) for label in labels], dtype=np.float64)
    akshara_lengths = np.asarray([len(split_aksharas(label)) for label in labels], dtype=np.float64)

    return {
        "unique_labels": len(set(labels)),
        "mean_chars": round(float(char_lengths.mean()), 3),
        "median_chars": round(float(np.median(char_lengths)), 3),
        "p95_chars": round(float(np.quantile(char_lengths, 0.95)), 3),
        "mean_aksharas": round(float(akshara_lengths.mean()), 3),
        "median_aksharas": round(float(np.median(akshara_lengths)), 3),
        "p95_aksharas": round(float(np.quantile(akshara_lengths, 0.95)), 3),
    }


def markdown_table(summaries: list[dict], include_label_stats: bool) -> str:
    lines = [
        "# Dataset Audit",
        "",
        f"Dataset: `{DATASET_NAME}`",
        "",
    ]
    if include_label_stats:
        lines.extend([
            "| Split | Samples | Unique labels | Mean chars | Median chars | P95 chars | Mean aksharas | Median aksharas | P95 aksharas |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summaries:
            lines.append(
                f"| {row['split']} | {row['num_samples']} | {row['unique_labels']} | "
                f"{row['mean_chars']:.3f} | {row['median_chars']:.3f} | {row['p95_chars']:.3f} | "
                f"{row['mean_aksharas']:.3f} | {row['median_aksharas']:.3f} | {row['p95_aksharas']:.3f} |"
            )
    else:
        lines.extend([
            "| Split | Samples | Approx. bytes |",
            "|---|---:|---:|",
        ])
        for row in summaries:
            size_mb = (row.get("num_bytes") or 0) / (1024 * 1024)
            lines.append(f"| {row['split']} | {row['num_samples']} | {size_mb:.1f} MB |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Paper 1 dataset splits.")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--include-label-stats", action="store_true",
                        help="Download/read full splits to compute label length stats.")
    args = parser.parse_args()

    metadata = {row["split"]: row for row in load_split_metadata()}
    summaries = [metadata[split] for split in args.splits if split in metadata]
    if args.include_label_stats:
        for row in summaries:
            row.update(summarize_labels(row["split"]))

    output_dir = Path(args.output_dir)
    save_results({"dataset": DATASET_NAME, "splits": summaries}, output_dir / "dataset_audit.json")

    markdown = markdown_table(summaries, args.include_label_stats)
    md_path = output_dir / "dataset_audit.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[dataset_audit] Wrote {md_path}")
    print(markdown)


if __name__ == "__main__":
    main()
