"""Miscellaneous utility helpers for configuration, logging, and batching."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
import yaml


LOGGER = logging.getLogger("dinov3_sam2")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class Batch:
    """Representation of a batch of prompts for SAM2."""

    boxes: List[List[float]]
    points: List[List[List[float]]]
    labels: List[List[int]]


def chunk_iterable(iterable: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def stack_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    mask_list = [mask.astype(np.uint8) for mask in masks]
    if not mask_list:
        return np.empty((0, 0, 0), dtype=np.uint8)
    height, width = mask_list[0].shape
    stacked = np.stack(mask_list, axis=0)
    assert stacked.shape[1:] == (height, width)
    return stacked


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise KeyError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def save_mask(path: str | os.PathLike[str], mask: np.ndarray, format_hint: str = "png") -> None:
    path = Path(path)
    if format_hint.lower() == "png":
        import cv2

        cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))
    elif format_hint.lower() == "npy":
        np.save(str(path), mask.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported mask format: {format_hint}")


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def gpu_memory_summary(device: torch.device | str) -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    dev = torch.device(device)
    allocated = torch.cuda.memory_allocated(dev) / 1024**2
    reserved = torch.cuda.memory_reserved(dev) / 1024**2
    return f"GPU memory - allocated: {allocated:.1f} MiB, reserved: {reserved:.1f} MiB"
