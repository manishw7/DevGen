"""
DevGen Framework — Smart Image Router

Analyzes an input image to determine if it contains a single
isolated character or a multi-character word/text.

Uses pure image processing (no ML) for fast routing (<1ms).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def classify_input_type(image: Image.Image) -> dict:
    """
    Determine if the image contains a single character or a word.

    Strategy:
    1. Convert to grayscale and binarize (Otsu-like threshold)
    2. Find connected components (ink blobs)
    3. Analyze bounding box aspect ratio and ink coverage
    4. Single char: ~square, 1 main blob, low ink coverage
       Word: wide, multiple blobs, higher ink coverage

    Returns:
        dict with "type" ("character" or "word"), "confidence", and debug info
    """
    # Work on grayscale numpy array
    gray = image.convert("L")
    arr = np.array(gray)

    # Binarize: ink pixels are dark, so invert
    # Adaptive threshold: use mean as threshold
    threshold = min(arr.mean() * 0.75, 200)
    binary = (arr < threshold).astype(np.uint8)

    # If almost no ink, probably empty — default to character
    ink_ratio = binary.sum() / binary.size
    if ink_ratio < 0.005:
        return {
            "type": "character",
            "confidence": 0.5,
            "reason": "very_low_ink",
            "ink_ratio": round(ink_ratio, 4),
            "blob_count": 0,
        }

    # Find bounding box of all ink
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return {"type": "character", "confidence": 0.5, "reason": "no_ink_found",
                "ink_ratio": 0.0, "blob_count": 0}

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    bbox_height = rmax - rmin + 1
    bbox_width = cmax - cmin + 1
    aspect_ratio = bbox_width / max(bbox_height, 1)

    # Count connected components using simple flood-fill
    blob_count = _count_blobs(binary, min_size=max(binary.size * 0.001, 10))

    # Ink density within bounding box
    bbox_area = bbox_height * bbox_width
    bbox_ink = binary[rmin:rmax+1, cmin:cmax+1].sum()
    bbox_density = bbox_ink / max(bbox_area, 1)

    # Decision logic
    is_character = True
    confidence = 0.5
    reason = "default"

    # Strong word signals
    if aspect_ratio > 2.5:
        is_character = False
        confidence = 0.95
        reason = "very_wide_aspect_ratio"
    elif aspect_ratio > 1.8 and blob_count >= 3:
        is_character = False
        confidence = 0.90
        reason = "wide_with_multiple_blobs"
    elif blob_count >= 4:
        is_character = False
        confidence = 0.85
        reason = "many_blobs"
    # Strong character signals
    elif aspect_ratio < 1.3 and blob_count <= 2:
        is_character = True
        confidence = 0.90
        reason = "square_single_blob"
    elif blob_count == 1 and aspect_ratio < 1.5:
        is_character = True
        confidence = 0.85
        reason = "single_blob_compact"
    # Ambiguous cases
    elif aspect_ratio < 1.75 and blob_count <= 2:
        is_character = True
        confidence = 0.65
        reason = "likely_character"
    elif aspect_ratio > 1.6:
        is_character = False
        confidence = 0.65
        reason = "likely_word"

    return {
        "type": "character" if is_character else "word",
        "confidence": round(confidence, 2),
        "reason": reason,
        "aspect_ratio": round(aspect_ratio, 3),
        "blob_count": blob_count,
        "ink_ratio": round(ink_ratio, 4),
        "bbox_density": round(bbox_density, 4),
    }


def _count_blobs(binary: np.ndarray, min_size: float = 10) -> int:
    """
    Count connected components in a binary image using iterative flood fill.
    Only counts blobs with at least `min_size` pixels.
    """
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                # Flood fill from this pixel
                size = _flood_fill(binary, visited, y, x, h, w)
                if size >= min_size:
                    count += 1

    return count


def _flood_fill(binary: np.ndarray, visited: np.ndarray,
                start_y: int, start_x: int, h: int, w: int) -> int:
    """Iterative flood fill. Returns size of the connected component."""
    stack = [(start_y, start_x)]
    size = 0

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x] or not binary[y, x]:
            continue

        visited[y, x] = True
        size += 1

        stack.append((y + 1, x))
        stack.append((y - 1, x))
        stack.append((y, x + 1))
        stack.append((y, x - 1))

    return size
