"""
Content tokenizers for the Devanagari handwriting LDM.

Two modes — the paper's central ablation:
  codepoint : one token per Unicode codepoint. Conjunct formation must be
              learned implicitly from co-occurring virama sequences.
  akshara   : one token per orthographic syllable (grapheme cluster), with
              codepoint FALLBACK for clusters unseen at vocab-build time, so
              out-of-vocabulary conjuncts remain representable. Whether this
              helps or hurts unseen-conjunct generalization is the question.

Vocabulary is built from the training split. Special ids:
  0 = PAD, 1 = NULL (classifier-free guidance null condition), 2 = UNK.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper1.experiments.akshara import split_aksharas  # noqa: E402

PAD_ID, NULL_ID, UNK_ID = 0, 1, 2
SPECIALS = ["<pad>", "<null>", "<unk>"]


class DevanagariTokenizer:
    def __init__(self, mode: str, vocab: dict[str, int], max_len: int):
        assert mode in ("codepoint", "akshara")
        self.mode = mode
        self.vocab = vocab
        self.max_len = max_len
        # codepoint entries double as the fallback space for akshara mode
        self.char_ids = {k: v for k, v in vocab.items() if len(k) == 1}

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    def units(self, text: str) -> list[str]:
        return split_aksharas(text) if self.mode == "akshara" else list(text)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for unit in self.units(text):
            if unit in self.vocab:
                ids.append(self.vocab[unit])
            elif self.mode == "akshara":
                # unseen cluster -> decompose to codepoints (compositional fallback)
                ids.extend(self.char_ids.get(ch, UNK_ID) for ch in unit)
            else:
                ids.append(UNK_ID)
        ids = ids[: self.max_len]
        return ids + [PAD_ID] * (self.max_len - len(ids))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mode": self.mode, "max_len": self.max_len, "vocab": self.vocab},
                       ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DevanagariTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["mode"], data["vocab"], data["max_len"])

    @classmethod
    def build(cls, texts: list[str], mode: str, min_freq: int = 2,
              max_len: int | None = None) -> "DevanagariTokenizer":
        from collections import Counter

        counts: Counter[str] = Counter()
        lengths: list[int] = []
        for text in texts:
            units = split_aksharas(text) if mode == "akshara" else list(text)
            counts.update(units)
            lengths.append(len(units))

        vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
        # always include every single codepoint seen (fallback space)
        chars = sorted({ch for t in texts for ch in t})
        for ch in chars:
            vocab.setdefault(ch, len(vocab))
        if mode == "akshara":
            for unit, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if freq >= min_freq and len(unit) > 1:
                    vocab.setdefault(unit, len(vocab))

        if max_len is None:
            lengths.sort()
            max_len = min(lengths[int(0.995 * len(lengths))] + 2, 32)
        return cls(mode, vocab, max_len)
