"""
DevGen Framework — Dataset Loader & Explorer
Provides endpoints to browse and sample from the IIIT-INDIC-HW-WORDS-Hindi dataset.
"""

import os
import io
import base64
import random
from pathlib import Path
from PIL import Image
from datasets import load_dataset, Dataset
from typing import Optional


# Default HuggingFace cache location
DATASET_NAME = "c3rl/IIIT-INDIC-HW-WORDS-Hindi"

_dataset_cache: dict = {}


def get_dataset(split: str = "train") -> Dataset:
    """Load a dataset split with caching."""
    if split not in _dataset_cache:
        print(f"[Dataset Loader] Loading '{DATASET_NAME}' split='{split}'...")
        ds = load_dataset(DATASET_NAME, split=split)
        _dataset_cache[split] = ds
        print(f"[Dataset Loader] Loaded {len(ds)} samples.")
    return _dataset_cache[split]


def get_dataset_info() -> dict:
    """Get metadata about all dataset splits."""
    info = {}
    for split in ["train", "validation", "test"]:
        try:
            ds = get_dataset(split)
            info[split] = {
                "num_samples": len(ds),
                "columns": ds.column_names,
            }
        except Exception as e:
            info[split] = {"error": str(e)}
    return info


def get_sample(split: str = "train", index: int = 0) -> dict:
    """Get a single sample from the dataset.
    
    Returns:
        dict with 'text', 'image_base64', and 'index'
    """
    ds = get_dataset(split)
    if index < 0 or index >= len(ds):
        raise IndexError(f"Index {index} out of range for split '{split}' (size={len(ds)})")
    
    sample = ds[index]
    image_b64 = _image_to_base64(sample["image"])
    
    return {
        "text": sample["text"],
        "image_base64": image_b64,
        "index": index,
        "split": split,
    }


def get_random_samples(split: str = "train", count: int = 10) -> list:
    """Get random samples from the dataset."""
    ds = get_dataset(split)
    indices = random.sample(range(len(ds)), min(count, len(ds)))
    samples = []
    for idx in indices:
        sample = ds[idx]
        samples.append({
            "text": sample["text"],
            "image_base64": _image_to_base64(sample["image"]),
            "index": idx,
            "split": split,
        })
    return samples


def get_paginated_samples(split: str = "train", page: int = 0,
                           page_size: int = 20) -> dict:
    """Get a paginated batch of samples."""
    ds = get_dataset(split)
    start = page * page_size
    end = min(start + page_size, len(ds))
    
    samples = []
    for idx in range(start, end):
        sample = ds[idx]
        samples.append({
            "text": sample["text"],
            "image_base64": _image_to_base64(sample["image"]),
            "index": idx,
        })
    
    return {
        "split": split,
        "page": page,
        "page_size": page_size,
        "total_samples": len(ds),
        "total_pages": (len(ds) + page_size - 1) // page_size,
        "samples": samples,
    }


def _image_to_base64(image) -> str:
    """Convert a PIL image or image dict to base64 string."""
    if isinstance(image, dict) and "bytes" in image:
        return base64.b64encode(image["bytes"]).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    elif isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    else:
        return ""
