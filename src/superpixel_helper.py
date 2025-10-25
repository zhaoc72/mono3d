"""Superpixel generation and feature aggregation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import cv2

from .utils import LOGGER


@dataclass
class SuperpixelConfig:
    """Configuration for superpixel generation."""
    
    method: str = "slic"  # slic, felzenszwalb, seeds
    n_segments: int = 1000  # 目标超像素数量
    compactness: float = 10.0  # SLIC 紧凑度
    sigma: float = 1.0  # 预处理高斯模糊
    
    # Felzenszwalb 参数
    scale: float = 100.0
    min_size: int = 50
    
    # SEEDS 参数
    num_levels: int = 4
    prior: int = 2


class SuperpixelGenerator:
    """Generate superpixels and aggregate features within them."""
    
    def __init__(self, config: SuperpixelConfig):
        self.config = config
    
    def generate_superpixels(self, image: np.ndarray) -> np.ndarray:
        """
        生成超像素分割
        
        Args:
            image: RGB 图像 [H, W, 3]
            
        Returns:
            超像素标签图 [H, W]，每个像素的值是其所属超像素的 ID
        """
        if self.config.method == "slic":
            return self._slic(image)
        elif self.config.method == "felzenszwalb":
            return self._felzenszwalb(image)
        elif self.config.method == "seeds":
            return self._seeds(image)
        else:
            raise ValueError(f"Unknown superpixel method: {self.config.method}")
    
    def _slic(self, image: np.ndarray) -> np.ndarray:
        """SLIC 超像素"""
        try:
            from skimage.segmentation import slic
            from skimage.color import rgb2lab
        except ImportError:
            raise ImportError("Please install scikit-image: pip install scikit-image")
        
        # 转换到 LAB 色彩空间
        lab_image = rgb2lab(image)
        
        # 生成超像素
        segments = slic(
            lab_image,
            n_segments=self.config.n_segments,
            compactness=self.config.compactness,
            sigma=self.config.sigma,
            start_label=0,
            channel_axis=2
        )
        
        LOGGER.info(f"SLIC generated {segments.max() + 1} superpixels")
        
        return segments
    
    def _felzenszwalb(self, image: np.ndarray) -> np.ndarray:
        """Felzenszwalb 超像素"""
        try:
            from skimage.segmentation import felzenszwalb
        except ImportError:
            raise ImportError("Please install scikit-image: pip install scikit-image")
        
        segments = felzenszwalb(
            image,
            scale=self.config.scale,
            sigma=self.config.sigma,
            min_size=self.config.min_size
        )
        
        LOGGER.info(f"Felzenszwalb generated {segments.max() + 1} superpixels")
        
        return segments
    
    def _seeds(self, image: np.ndarray) -> np.ndarray:
        """SEEDS 超像素 (使用 OpenCV)"""
        if not hasattr(cv2.ximgproc, 'createSuperpixelSEEDS'):
            raise RuntimeError(
                "OpenCV SEEDS not available. Install opencv-contrib-python: "
                "pip install opencv-contrib-python"
            )
        
        height, width = image.shape[:2]
        
        # 转换为 BGR (OpenCV 格式)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 创建 SEEDS 对象
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            width, height, 3,
            self.config.n_segments,
            self.config.num_levels,
            self.config.prior
        )
        
        # 迭代生成
        seeds.iterate(bgr_image, num_iterations=10)
        
        # 获取标签
        segments = seeds.getLabels()
        
        LOGGER.info(f"SEEDS generated {seeds.getNumberOfSuperpixels()} superpixels")
        
        return segments
    
    def aggregate_features_by_superpixels(
        self,
        patch_features: np.ndarray,
        superpixel_labels: np.ndarray,
        image_shape: Tuple[int, int],
        patch_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在超像素内聚合 patch 特征
        
        Args:
            patch_features: [N_patches, D] DINOv3 patch 特征
            superpixel_labels: [H, W] 超像素标签
            image_shape: (height, width) 原图尺寸
            
        Returns:
            (superpixel_features, superpixel_ids)
            - superpixel_features: [N_superpixels, D] 每个超像素的平均特征
            - superpixel_ids: [N_superpixels] 超像素 ID
        """
        # 计算 patch grid 大小
        features = np.asarray(patch_features, dtype=np.float32)
        if patch_shape is None:
            n_patches = features.shape[0]
            grid_side = int(round(np.sqrt(n_patches)))
            if grid_side * grid_side != n_patches:
                raise ValueError(
                    f"Patch features ({n_patches}) cannot form a square grid; "
                    "please provide patch_shape explicitly"
                )
            patch_h, patch_w = grid_side, grid_side
        else:
            patch_h, patch_w = patch_shape
            if patch_h * patch_w != features.shape[0]:
                raise ValueError(
                    f"Provided patch_shape {patch_shape} is incompatible with features "
                    f"({features.shape[0]})"
                )

        # 上采样 patch 特征到原图尺寸
        patch_map = features.reshape(patch_h, patch_w, -1)
        height, width = image_shape

        # 使用最近邻插值上采样每个特征通道
        upsampled_features = np.zeros((height, width, features.shape[1]), dtype=np.float32)
        for i in range(features.shape[1]):
            upsampled_features[:, :, i] = cv2.resize(
                patch_map[:, :, i],
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 在每个超像素内聚合特征
        unique_labels = np.unique(superpixel_labels)
        n_superpixels = len(unique_labels)
        feature_dim = features.shape[1]
        
        superpixel_features = np.zeros((n_superpixels, feature_dim), dtype=np.float32)
        
        for idx, label in enumerate(unique_labels):
            mask = superpixel_labels == label
            # 取该超像素内所有像素特征的平均
            superpixel_features[idx] = upsampled_features[mask].mean(axis=0)
        
        LOGGER.info(f"Aggregated features for {n_superpixels} superpixels")
        
        return superpixel_features, unique_labels
    
    def create_adjacency_graph(
        self,
        superpixel_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建超像素邻接图
        
        Args:
            superpixel_labels: [H, W] 超像素标签
            
        Returns:
            (edges, edge_weights)
            - edges: [N_edges, 2] 边的端点 (i, j)
            - edge_weights: [N_edges] 边的权重（边界长度）
        """
        from collections import defaultdict
        
        height, width = superpixel_labels.shape
        
        # 统计相邻超像素对
        adjacency_count = defaultdict(int)
        
        # 水平方向
        for i in range(height):
            for j in range(width - 1):
                label1 = superpixel_labels[i, j]
                label2 = superpixel_labels[i, j + 1]
                if label1 != label2:
                    edge = tuple(sorted([label1, label2]))
                    adjacency_count[edge] += 1
        
        # 垂直方向
        for i in range(height - 1):
            for j in range(width):
                label1 = superpixel_labels[i, j]
                label2 = superpixel_labels[i + 1, j]
                if label1 != label2:
                    edge = tuple(sorted([label1, label2]))
                    adjacency_count[edge] += 1
        
        # 转换为数组
        edges = np.array(list(adjacency_count.keys()), dtype=np.int32)
        edge_weights = np.array(list(adjacency_count.values()), dtype=np.float32)
        
        # 归一化权重
        edge_weights = edge_weights / edge_weights.max()
        
        LOGGER.info(f"Created adjacency graph with {len(edges)} edges")
        
        return edges, edge_weights


def visualize_superpixels(
    image: np.ndarray,
    superpixel_labels: np.ndarray,
    output_path: str
) -> None:
    """
    可视化超像素边界
    
    Args:
        image: RGB 图像
        superpixel_labels: 超像素标签
        output_path: 输出路径
    """
    try:
        from skimage.segmentation import mark_boundaries
    except ImportError:
        LOGGER.warning("Cannot visualize superpixels: scikit-image not installed")
        return
    
    # 绘制边界
    marked = mark_boundaries(image, superpixel_labels, color=(1, 0, 0), mode='thick')
    marked = (marked * 255).astype(np.uint8)
    
    # 保存
    cv2.imwrite(output_path, cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    LOGGER.info(f"Superpixel visualization saved to {output_path}")