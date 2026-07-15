"""
Build the three conjunct-stratified evaluation vocabularies for Paper 2:

  iv.txt                  in-vocabulary words (sampled from train split)
  oov_seen.txt            pseudo-words: unseen word forms composed ONLY of
                          aksharas that appear in training
  oov_unseen_conjunct.txt pseudo-words containing at least one conjunct
                          cluster that never appears in training

Pseudo-words are constructed by recombining aksharas, so every generated
form is orthographically valid Devanagari even when it is not a lexical
word — the generator is judged on rendering fidelity, not lexicality.

Usage:
    python -m paper2.experiments.build_eval_vocab \
        --parquet-dir data/iiit_hindi_parquet --count 200
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper1.experiments.akshara import (  # noqa: E402
    VIRAMA,
    is_conjunct,
    is_consonant,
    split_aksharas,
)
from paper1.experiments.common import normalize_text  # noqa: E402

# All base consonants for constructing unseen conjuncts
_ALL_CONSONANTS = [chr(c) for c in range(0x0915, 0x093A)]
_COMMON_MATRAS = ["", "ा", "ि", "ी", "े", "ो", "ु", "ू"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Paper 2 eval vocabularies.")
    parser.add_argument("--parquet-dir", default="data/iiit_hindi_parquet")
    parser.add_argument("--count", type=int, default=200, help="Words per stratum.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="paper2/experiments/vocab")
    return parser.parse_args()


def load_train_words(parquet_dir: str) -> list[str]:
    from datasets import load_dataset

    files = sorted(str(p) for p in Path(parquet_dir).glob("train-*.parquet"))
    dataset = load_dataset("parquet", data_files=files, split="train")
    return [normalize_text(t) for t in dataset["text"] if str(t).strip()]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = load_train_words(args.parquet_dir)
    train_forms = set(words)
    akshara_counts: Counter[str] = Counter()
    conjunct_set: set[str] = set()
    for word in train_forms:
        for cluster in split_aksharas(word):
            akshara_counts[cluster] += 1
            if is_conjunct(cluster):
                conjunct_set.add(cluster)
    def is_devanagari_cluster(cluster: str) -> bool:
        return bool(cluster.strip()) and all(0x0900 <= ord(ch) <= 0x097F for ch in cluster)

    seen_aksharas = [a for a, c in akshara_counts.items() if c >= 3 and is_devanagari_cluster(a)]
    print(f"[vocab] {len(train_forms)} unique train words, "
          f"{len(akshara_counts)} akshara types, {len(conjunct_set)} conjunct types")

    # --- Stratum 1: in-vocabulary ---
    iv = rng.sample(sorted(train_forms), args.count)

    # --- Stratum 2: OOV forms from seen aksharas ---
    oov_seen: set[str] = set()
    attempts = 0
    while len(oov_seen) < args.count and attempts < args.count * 200:
        attempts += 1
        length = rng.randint(2, 5)
        form = "".join(rng.choice(seen_aksharas) for _ in range(length))
        if form not in train_forms and form not in oov_seen:
            oov_seen.add(form)

    # --- Stratum 3: OOV forms with unseen conjuncts ---
    # Construct C1+virama+C2 clusters absent from training, embed each in a
    # word otherwise made of frequent seen aksharas.
    unseen_conjuncts: list[str] = []
    frequent = [a for a, c in akshara_counts.most_common(300)
                if not is_conjunct(a) and is_devanagari_cluster(a)][:200]
    for c1 in _ALL_CONSONANTS:
        for c2 in _ALL_CONSONANTS:
            for matra in _COMMON_MATRAS:
                cluster = c1 + VIRAMA + c2 + matra
                if cluster not in conjunct_set and is_consonant(c1) and is_consonant(c2):
                    unseen_conjuncts.append(cluster)
    rng.shuffle(unseen_conjuncts)
    oov_unseen: set[str] = set()
    idx = 0
    while len(oov_unseen) < args.count and idx < len(unseen_conjuncts):
        cluster = unseen_conjuncts[idx]
        idx += 1
        prefix = rng.choice(frequent) if rng.random() < 0.7 else ""
        suffix = rng.choice(frequent)
        form = prefix + cluster + suffix
        if form not in train_forms:
            oov_unseen.add(form)

    for name, vocab in [("iv", iv), ("oov_seen", sorted(oov_seen)),
                        ("oov_unseen_conjunct", sorted(oov_unseen))]:
        path = out_dir / f"{name}.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        print(f"[vocab] {path}: {len(vocab)} words")


if __name__ == "__main__":
    main()
