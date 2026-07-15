"""
Script-aware error analysis over a results JSON produced by evaluate.py.

Answers the Devanagari-specific questions no Latin-script HTR paper covers:
  - How much error comes from conjunct clusters vs simple aksharas?
  - Are matras (dependent vowels) or nasalization signs the dominant failure?
  - Which akshara confusion pairs are most frequent?
  - Does error grow with word length / conjunct density?

Usage:
    python -m paper1.experiments.error_analysis \
        --results paper1/experiments/results/lora_r16_legacy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper1.experiments.akshara import (  # noqa: E402
    align_aksharas,
    is_conjunct,
    split_aksharas,
)

CATEGORY_LABELS = {
    "conjunct_substitution": "Conjunct substitution",
    "matra_error": "Matra (dependent vowel) error",
    "sign_error": "Nasalization/visarga sign error",
    "base_substitution": "Base consonant substitution",
    "insertion": "Insertion",
    "insertion_conjunct": "Insertion (conjunct)",
    "deletion": "Deletion",
    "deletion_conjunct": "Deletion (conjunct)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Devanagari error analysis of eval results.")
    parser.add_argument("--results", required=True, help="Path to evaluate.py results JSON.")
    parser.add_argument("--top-confusions", type=int, default=25)
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default: <results>_analysis.md).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    with results_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    samples = data["samples"]
    run_name = data["summary"]["run_name"]

    category_counts: Counter[str] = Counter()
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    total_ref_aksharas = 0
    conjunct_ref_total = 0
    conjunct_ref_errors = 0
    simple_ref_total = 0
    simple_ref_errors = 0
    # error rate bucketed by word length (in aksharas) and conjunct density
    length_buckets: dict[str, list[float]] = {"1-2": [], "3-4": [], "5-6": [], "7+": []}
    conjunct_buckets: dict[str, list[float]] = {"0 conjuncts": [], "1 conjunct": [], "2+ conjuncts": []}

    for sample in samples:
        ref_clusters = split_aksharas(sample["reference"])
        pred_clusters = split_aksharas(sample["prediction"])
        edits = align_aksharas(ref_clusters, pred_clusters)

        total_ref_aksharas += len(ref_clusters)
        n_conjuncts = sum(1 for c in ref_clusters if is_conjunct(c))
        conjunct_ref_total += n_conjuncts
        simple_ref_total += len(ref_clusters) - n_conjuncts

        for edit in edits:
            category_counts[edit.category()] += 1
            if edit.ref:
                if is_conjunct(edit.ref):
                    conjunct_ref_errors += 1
                else:
                    simple_ref_errors += 1
            if edit.op == "sub":
                confusion_pairs[(edit.ref, edit.pred)] += 1

        n = len(ref_clusters)
        length_key = "1-2" if n <= 2 else "3-4" if n <= 4 else "5-6" if n <= 6 else "7+"
        length_buckets[length_key].append(sample["aer"])
        conjunct_key = "0 conjuncts" if n_conjuncts == 0 else "1 conjunct" if n_conjuncts == 1 else "2+ conjuncts"
        conjunct_buckets[conjunct_key].append(sample["aer"])

    total_edits = sum(category_counts.values())

    lines: list[str] = []
    lines.append(f"# Devanagari error analysis — `{run_name}`")
    lines.append("")
    lines.append(f"Samples: {len(samples)} — total reference aksharas: {total_ref_aksharas} — "
                 f"total akshara-level edits: {total_edits}")
    lines.append("")

    lines.append("## Error category breakdown")
    lines.append("")
    lines.append("| Category | Count | % of errors |")
    lines.append("|---|---:|---:|")
    for key, count in category_counts.most_common():
        share = 100.0 * count / total_edits if total_edits else 0.0
        lines.append(f"| {CATEGORY_LABELS.get(key, key)} | {count} | {share:.1f}% |")
    lines.append("")

    lines.append("## Conjunct vs simple akshara error rate")
    lines.append("")
    lines.append("| Akshara type | Occurrences | Errored | Error rate |")
    lines.append("|---|---:|---:|---:|")
    conj_rate = conjunct_ref_errors / conjunct_ref_total if conjunct_ref_total else 0.0
    simple_rate = simple_ref_errors / simple_ref_total if simple_ref_total else 0.0
    lines.append(f"| Conjunct (contains virama-joined consonants) | {conjunct_ref_total} | {conjunct_ref_errors} | {conj_rate:.4f} |")
    lines.append(f"| Simple | {simple_ref_total} | {simple_ref_errors} | {simple_rate:.4f} |")
    lines.append("")

    lines.append("## Top akshara confusion pairs (reference → prediction)")
    lines.append("")
    lines.append("| Reference | Prediction | Count |")
    lines.append("|---|---|---:|")
    for (ref, pred), count in confusion_pairs.most_common(args.top_confusions):
        lines.append(f"| {ref} | {pred} | {count} |")
    lines.append("")

    lines.append("## Akshara error rate by word length (aksharas)")
    lines.append("")
    lines.append("| Length | Words | Mean AER |")
    lines.append("|---|---:|---:|")
    for key, values in length_buckets.items():
        mean = sum(values) / len(values) if values else 0.0
        lines.append(f"| {key} | {len(values)} | {mean:.4f} |")
    lines.append("")

    lines.append("## Akshara error rate by conjunct count")
    lines.append("")
    lines.append("| Conjuncts in word | Words | Mean AER |")
    lines.append("|---|---:|---:|")
    for key, values in conjunct_buckets.items():
        mean = sum(values) / len(values) if values else 0.0
        lines.append(f"| {key} | {len(values)} | {mean:.4f} |")
    lines.append("")

    output_path = Path(args.output) if args.output else results_path.with_name(
        results_path.stem + "_analysis.md"
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[error_analysis] Wrote {output_path}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
