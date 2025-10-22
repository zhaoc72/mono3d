"""Mask post-processing routines for quality improvement."""
from __future__ import annotations

from typing import Iterable, List, Optional

import cv2
import numpy as np


def select_masks_by_area(masks: Iterable[np.ndarray], min_area: int) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    for mask in masks:
        area = float(mask.astype(np.uint8).sum())
        if area >= min_area:
            selected.append(mask)
    return selected


def select_largest_mask(masks: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    best_mask = None
    best_area = -1
    for mask in masks:
        area = float(mask.astype(np.uint8).sum())
        if area > best_area:
            best_area = area
            best_mask = mask
    return best_mask


def smooth_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return smoothed.astype(bool)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((radius, radius), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated.astype(bool)


def apply_postprocessing(mask: np.ndarray, kernel_size: int, dilation_radius: int) -> np.ndarray:
    smoothed = smooth_mask(mask, kernel_size)
    dilated = dilate_mask(smoothed, dilation_radius)
    return dilated
