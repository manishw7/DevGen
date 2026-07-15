"""
FID between generated word images and real test-split word images.

Uses clean-fid (pip install clean-fid). Real images are exported once from
the parquet test split to a folder; FID is then folder-vs-folder.

Usage:
    python -m paper2.experiments.compute_fid \
        --generated-dir data_synth/eval_iv/images \
        --parquet-dir data/iiit_hindi_parquet --real-limit 5000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REAL_CACHE = Path(__file__).resolve().parent / "real_test_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FID for generated handwriting.")
    parser.add_argument("--generated-dir", required=True, help="Folder of generated PNGs.")
    parser.add_argument("--parquet-dir", default="data/iiit_hindi_parquet")
    parser.add_argument("--real-limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=256, help="Resize both sides to this square.")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def export_real_images(parquet_dir: str, limit: int, size: int) -> Path:
    out_dir = REAL_CACHE / f"n{limit}_s{size}"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= limit:
        print(f"[fid] Reusing cached real images at {out_dir}")
        return out_dir
    from datasets import load_dataset

    files = sorted(str(p) for p in Path(parquet_dir).glob("test-*.parquet"))
    dataset = load_dataset("parquet", data_files=files, split=f"train[:{limit}]")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(dataset):
        img = sample["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(
            out_dir / f"real_{i:06d}.png"
        )
    print(f"[fid] Exported {limit} real images to {out_dir}")
    return out_dir


def main() -> None:
    args = parse_args()
    from cleanfid import fid as cleanfid

    real_dir = export_real_images(args.parquet_dir, args.real_limit, args.size)

    # Resize generated copies to the same square for a fair comparison
    generated_dir = Path(args.generated_dir)
    resized_dir = generated_dir.parent / f"{generated_dir.name}_fid{args.size}"
    resized_dir.mkdir(exist_ok=True)
    pngs = sorted(generated_dir.glob("*.png"))
    for path in pngs:
        target = resized_dir / path.name
        if not target.exists():
            Image.open(path).convert("RGB").resize(
                (args.size, args.size), Image.Resampling.LANCZOS
            ).save(target)

    score = cleanfid.compute_fid(str(real_dir), str(resized_dir))
    run_name = args.run_name or generated_dir.parent.name
    print(f"[fid] {run_name}: FID = {score:.2f}  "
          f"({len(pngs)} generated vs {args.real_limit} real @ {args.size}px)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"fid_{run_name}.json"
    out.write_text(json.dumps({
        "run_name": run_name,
        "fid": round(score, 4),
        "n_generated": len(pngs),
        "n_real": args.real_limit,
        "size": args.size,
    }, indent=2), encoding="utf-8")
    print(f"[fid] Wrote {out}")


if __name__ == "__main__":
    main()
