"""
Aggregate results/*.json from evaluate.py into publication tables
(Markdown for inspection, LaTeX booktabs for the paper).

Grouping is by run-name convention:
    base_checkpoint, full_ft, lora_r{R}_{modules}     -> main + ablation tables
    lora_r16_legacy_synth{PCT}                        -> augmentation table

Usage:
    python -m paper1.experiments.make_tables
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_TABLE = REPO_ROOT / "paper1" / "paper" / "tables" / "tables.tex"


def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "summary" in data:
            summaries.append(data["summary"])
    return summaries


def fmt_params(count) -> str:
    if count is None:
        return "—"
    return f"{count / 1e6:.2f}M"


def fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def markdown_table(summaries: list[dict]) -> str:
    lines = [
        "| Run | Trainable params | CER ↓ | CER 95% CI | Akshara ER ↓ | Word acc. ↑ |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['run_name']} | {fmt_params(s.get('trainable_params'))} | "
            f"{s['cer']:.4f} | {fmt_ci(s['cer_ci95'])} | {s['aer']:.4f} | {s['word_accuracy']:.4f} |"
        )
    return "\n".join(lines)


def latex_table(summaries: list[dict], caption: str, label: str) -> str:
    rows = []
    for s in summaries:
        name = s["run_name"].replace("_", r"\_")
        rows.append(
            f"    {name} & {fmt_params(s.get('trainable_params'))} & "
            f"{s['cer']:.4f} & {s['aer']:.4f} & {s['word_accuracy']:.4f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{tabular}{lrrrr}\n"
        "    \\toprule\n"
        "    Run & Trainable & CER $\\downarrow$ & AER $\\downarrow$ & WAcc $\\uparrow$ \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build result tables from eval JSONs.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "tables"))
    parser.add_argument(
        "--paper-table-path",
        default=str(DEFAULT_PAPER_TABLE),
        help="Optional LaTeX table mirror used by paper1/paper/main.tex; pass '' to skip.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"[make_tables] No results in {results_dir} — run evaluate.py first.")
        return

    synth = [s for s in summaries if re.search(r"synth", s["run_name"])]
    main_runs = [s for s in summaries if s not in synth]
    main_runs.sort(key=lambda s: s["cer"])
    synth.sort(key=lambda s: s["run_name"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_parts = ["# Paper 1 result tables", "", "## Main comparison / LoRA ablation", "",
                markdown_table(main_runs), ""]
    tex_parts = [latex_table(
        main_runs,
        "Test-split results on IIIT-INDIC-HW-WORDS-Hindi. CER: character error rate "
        "(NFC codepoints); AER: akshara error rate; WAcc: word recognition accuracy.",
        "tab:main",
    )]
    if synth:
        md_parts += ["## Synthetic data augmentation", "", markdown_table(synth), ""]
        tex_parts.append(latex_table(
            synth,
            "Effect of LDM-generated synthetic training data on recognition.",
            "tab:synth",
        ))

    md_text = "\n".join(md_parts)
    tex_text = "\n\n".join(tex_parts)
    (output_dir / "tables.md").write_text(md_text, encoding="utf-8")
    (output_dir / "tables.tex").write_text(tex_text, encoding="utf-8")
    if args.paper_table_path:
        paper_table_path = Path(args.paper_table_path)
        paper_table_path.parent.mkdir(parents=True, exist_ok=True)
        paper_table_path.write_text(tex_text, encoding="utf-8")
    print(f"[make_tables] Wrote {output_dir}/tables.md and tables.tex "
          f"({len(main_runs)} main runs, {len(synth)} augmentation runs)")
    if args.paper_table_path:
        print(f"[make_tables] Mirrored LaTeX table to {args.paper_table_path}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
