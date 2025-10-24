"""Mask post-processing routines for quality improvement."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

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


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个掩码的 IoU
    
    Args:
        mask1: 二值掩码 [H, W]
        mask2: 二值掩码 [H, W]
        
    Returns:
        IoU 值 (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def iou_nms(
    masks: List[np.ndarray],
    scores: Optional[List[float]] = None,
    iou_threshold: float = 0.6
) -> Tuple[List[np.ndarray], List[int]]:
    """
    使用 IoU-NMS 去重掩码
    
    Args:
        masks: 掩码列表
        scores: 每个掩码的得分（越高越好），如果为 None 则按面积排序
        iou_threshold: IoU 阈值，超过此值的掩码会被抑制
        
    Returns:
        (保留的掩码列表, 保留的索引列表)
    """
    if not masks:
        return [], []
    
    # 如果没有提供得分，使用面积作为得分
    if scores is None:
        scores = [float(mask.astype(np.uint8).sum()) for mask in masks]
    
    # 按得分降序排序
    indices = np.argsort(scores)[::-1].tolist()
    
    keep_indices = []
    
    while indices:
        # 选择得分最高的
        current_idx = indices.pop(0)
        keep_indices.append(current_idx)
        current_mask = masks[current_idx]
        
        # 移除与当前掩码 IoU 过高的掩码
        remaining_indices = []
        for idx in indices:
            iou = compute_iou(current_mask, masks[idx])
            if iou < iou_threshold:
                remaining_indices.append(idx)
        
        indices = remaining_indices
    
    # 返回保留的掩码
    keep_masks = [masks[i] for i in keep_indices]
    
    return keep_masks, keep_indices


def nms_with_objectness(
    masks: List[np.ndarray],
    objectness_scores: List[float],
    sam2_confidence: Optional[List[float]] = None,
    iou_threshold: float = 0.6,
    objectness_weight: float = 0.5,
    confidence_weight: float = 0.3,
    area_weight: float = 0.2
) -> Tuple[List[np.ndarray], List[int]]:
    """
    使用对象性、SAM2 置信度和面积的综合评分进行 NMS
    
    Args:
        masks: 掩码列表
        objectness_scores: 对象性评分 (0-1)
        sam2_confidence: SAM2 置信度评分 (0-1)，可选
        iou_threshold: IoU 阈值
        objectness_weight: 对象性权重
        confidence_weight: SAM2 置信度权重
        area_weight: 面积权重
        
    Returns:
        (保留的掩码列表, 保留的索引列表)
    """
    if not masks:
        return [], []
    
    # 计算面积评分（归一化）
    areas = [float(mask.astype(np.uint8).sum()) for mask in masks]
    max_area = max(areas) if areas else 1.0
    area_scores = [a / max_area for a in areas]
    
    # 计算综合评分
    combined_scores = []
    for i in range(len(masks)):
        score = objectness_weight * objectness_scores[i] + area_weight * area_scores[i]
        
        if sam2_confidence is not None:
            score += confidence_weight * sam2_confidence[i]
        else:
            # 如果没有 SAM2 置信度，重新分配权重
            score = (
                (objectness_weight / (objectness_weight + area_weight)) * objectness_scores[i] +
                (area_weight / (objectness_weight + area_weight)) * area_scores[i]
            )
        
        combined_scores.append(score)
    
    # 使用综合评分进行 NMS
    return iou_nms(masks, scores=combined_scores, iou_threshold=iou_threshold)


def filter_by_combined_score(
    masks: List[np.ndarray],
    objectness_scores: List[float],
    min_objectness: float = 0.3,
    min_area: int = 100
) -> Tuple[List[np.ndarray], List[int]]:
    """
    使用对象性和面积综合过滤掩码
    
    Args:
        masks: 掩码列表
        objectness_scores: 对象性评分
        min_objectness: 最小对象性阈值
        min_area: 最小面积阈值
        
    Returns:
        (过滤后的掩码列表, 保留的索引列表)
    """
    keep_indices = []
    keep_masks = []
    
    for i, (mask, obj_score) in enumerate(zip(masks, objectness_scores)):
        area = float(mask.astype(np.uint8).sum())
        
        if obj_score >= min_objectness and area >= min_area:
            keep_indices.append(i)
            keep_masks.append(mask)
    
    return keep_masks, keep_indices