"""
Devanagari akshara (orthographic syllable / grapheme cluster) segmentation
and script-aware error categorization.

An akshara is the perceptual writing unit of Devanagari:
    (C halant)* C (nukta)? (matra)? (sign)*   e.g. क्ष्मी = क + ् + ष + ् + म + ी
    V (sign)*                                 independent vowel, e.g. आँ
Codepoint-level CER treats क्ष (3 codepoints) and क (1) asymmetrically;
akshara-level metrics weight them equally, matching how readers perceive errors.
"""

from __future__ import annotations

from dataclasses import dataclass

VIRAMA = "्"  # ्
NUKTA = "़"   # ़
ZWJ_ZWNJ = {"‌", "‍"}

# Unicode Devanagari block ranges
_CONSONANTS = set(
    [chr(c) for c in range(0x0915, 0x093A)]  # क..ह
    + [chr(c) for c in range(0x0958, 0x0960)]  # nukta consonants क़..य़
    + ["ॻ", "ॼ", "ॾ", "ॿ"]  # rare extensions
)
_INDEPENDENT_VOWELS = {chr(c) for c in range(0x0904, 0x0915)}  # ऄ..औ
_MATRAS = {chr(c) for c in range(0x093E, 0x094D)} | {"ॢ", "ॣ", "ऺ", "ऻ", "ॎ", "ॏ"}
_SIGNS = {"ँ", "ं", "ः"}  # candrabindu, anusvara, visarga
_DIGITS = {chr(c) for c in range(0x0966, 0x0970)}  # ०..९


def is_consonant(ch: str) -> bool:
    return ch in _CONSONANTS


def split_aksharas(text: str) -> list[str]:
    """Segment NFC-normalized Devanagari text into akshara clusters.
    Non-Devanagari characters become single-character clusters."""
    clusters: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if is_consonant(ch):
            j = i + 1
            while j < n and text[j] == NUKTA:
                j += 1
            # consume (halant [ZWJ] consonant)* chains — conjuncts
            while (
                j < n
                and text[j] == VIRAMA
                and (
                    (j + 1 < n and is_consonant(text[j + 1]))
                    or (j + 2 < n and text[j + 1] in ZWJ_ZWNJ and is_consonant(text[j + 2]))
                )
            ):
                j += 2 if is_consonant(text[j + 1]) else 3
                while j < n and text[j] == NUKTA:
                    j += 1
            # word-final dead consonant (trailing halant)
            if j < n and text[j] == VIRAMA and (j + 1 == n or not is_consonant(text[j + 1])):
                j += 1
            if j < n and text[j] in _MATRAS:
                j += 1
            while j < n and text[j] in _SIGNS:
                j += 1
            clusters.append(text[i:j])
            i = j
        elif ch in _INDEPENDENT_VOWELS:
            j = i + 1
            while j < n and text[j] in _SIGNS:
                j += 1
            clusters.append(text[i:j])
            i = j
        else:
            clusters.append(ch)
            i += 1
    return clusters


def is_conjunct(cluster: str) -> bool:
    """True if the cluster contains a consonant-joining virama (e.g. क्ष, त्र, स्थ)."""
    for k, ch in enumerate(cluster):
        if ch == VIRAMA and k + 1 < len(cluster):
            nxt = cluster[k + 1]
            if is_consonant(nxt) or nxt in ZWJ_ZWNJ:
                return True
    return False


def has_matra(cluster: str) -> bool:
    return any(ch in _MATRAS for ch in cluster)


def has_sign(cluster: str) -> bool:
    return any(ch in _SIGNS for ch in cluster)


def base_consonants(cluster: str) -> str:
    """The consonant/vowel skeleton of a cluster, stripped of matras and signs."""
    return "".join(ch for ch in cluster if is_consonant(ch) or ch in _INDEPENDENT_VOWELS or ch == VIRAMA)


@dataclass
class AksharaEdit:
    op: str  # "sub" | "ins" | "del"
    ref: str  # reference cluster ("" for insertions)
    pred: str  # predicted cluster ("" for deletions)

    def category(self) -> str:
        """Script-aware error category, checked most-specific first."""
        ref, pred = self.ref, self.pred
        if self.op == "ins":
            return "insertion_conjunct" if is_conjunct(pred) else "insertion"
        if self.op == "del":
            return "deletion_conjunct" if is_conjunct(ref) else "deletion"
        # substitution subtypes
        if is_conjunct(ref) or is_conjunct(pred):
            return "conjunct_substitution"
        if base_consonants(ref) == base_consonants(pred):
            if has_sign(ref) != has_sign(pred) and has_matra(ref) == has_matra(pred):
                return "sign_error"  # anusvara/candrabindu/visarga only
            return "matra_error"  # same skeleton, different vowel marking
        return "base_substitution"


def align_aksharas(ref_clusters: list[str], pred_clusters: list[str]) -> list[AksharaEdit]:
    """Levenshtein alignment over cluster sequences; returns only edit ops."""
    m, n = len(ref_clusters), len(pred_clusters)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_clusters[i - 1] == pred_clusters[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edits: list[AksharaEdit] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref_clusters[i - 1] == pred_clusters[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            edits.append(AksharaEdit("sub", ref_clusters[i - 1], pred_clusters[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(AksharaEdit("del", ref_clusters[i - 1], ""))
            i -= 1
        else:
            edits.append(AksharaEdit("ins", "", pred_clusters[j - 1]))
            j -= 1
    edits.reverse()
    return edits
