"""Density-based clustering methods for instance segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .utils import LOGGER


@dataclass
class DensityClusterConfig:
    """Configuration for density-based clustering."""
    
    method: str = "hdbscan"  # hdbscan, meanshift, dbscan
    
    # HDBSCAN 参数
    min_cluster_size: int = 50
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.0
    
    # MeanShift 参数
    bandwidth: Optional[float] = None  # None = 自动估计
    
    # DBSCAN 参数
    eps: float = 0.5
    dbscan_min_samples: int = 5


class DensityClusterer:
    """Density-based clustering for separating instances."""
    
    def __init__(self, config: DensityClusterConfig):
        self.config = config
    
    def cluster(self, features: np.ndarray) -> np.ndarray:
        """
        执行密度聚类
        
        Args:
            features: [N, D] 特征矩阵
            
        Returns:
            labels: [N] 聚类标签（-1 表示噪声）
        """
        if self.config.method == "hdbscan":
            return self._hdbscan_clustering(features)
        elif self.config.method == "meanshift":
            return self._meanshift_clustering(features)
        elif self.config.method == "dbscan":
            return self._dbscan_clustering(features)
        else:
            raise ValueError(f"Unknown density clustering method: {self.config.method}")
    
    def _hdbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """HDBSCAN 聚类"""
        try:
            import hdbscan
        except ImportError:
            LOGGER.warning(
                "HDBSCAN not available; falling back to MeanShift for density clustering"
            )
            return self._meanshift_clustering(features)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric='euclidean',
            core_dist_n_jobs=-1
        )
        
        labels = clusterer.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        LOGGER.info(
            f"HDBSCAN: {len(features)} points -> {n_clusters} clusters, "
            f"{n_noise} noise points"
        )
        
        return labels
    
    def _meanshift_clustering(self, features: np.ndarray) -> np.ndarray:
        """MeanShift 聚类"""
        try:
            from sklearn.cluster import MeanShift, estimate_bandwidth
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        # 估计 bandwidth（如果未提供）
        bandwidth = self.config.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(
                features,
                quantile=0.2,
                n_samples=min(500, len(features))
            )
            LOGGER.info(f"Estimated bandwidth: {bandwidth:.4f}")
        
        # MeanShift 聚类
        ms = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            min_bin_freq=5
        )
        
        labels = ms.fit_predict(features)
        
        n_clusters = len(np.unique(labels))
        
        LOGGER.info(f"MeanShift: {len(features)} points -> {n_clusters} clusters")
        
        return labels
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """DBSCAN 聚类"""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        db = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.dbscan_min_samples,
            metric='euclidean',
            n_jobs=-1
        )
        
        labels = db.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        LOGGER.info(
            f"DBSCAN: {len(features)} points -> {n_clusters} clusters, "
            f"{n_noise} noise points"
        )
        
        return labels


def handle_noise_points(
    labels: np.ndarray,
    features: np.ndarray,
    method: str = "nearest"
) -> np.ndarray:
    """
    处理噪声点（标签为 -1 的点）
    
    Args:
        labels: 聚类标签
        features: 特征向量
        method: 处理方法 ("nearest", "remove")
        
    Returns:
        处理后的标签
    """
    if method == "remove":
        # 保持噪声标签不变
        return labels
    
    elif method == "nearest":
        # 将噪声点分配到最近的聚类
        noise_mask = labels == -1
        if not noise_mask.any():
            return labels
        
        valid_mask = ~noise_mask
        valid_labels = np.unique(labels[valid_mask])
        
        # 计算每个聚类的质心
        centroids = []
        for label in valid_labels:
            cluster_mask = labels == label
            centroid = features[cluster_mask].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # 为每个噪声点分配最近的聚类
        new_labels = labels.copy()
        noise_features = features[noise_mask]
        
        for i, noise_feat in enumerate(noise_features):
            distances = np.linalg.norm(centroids - noise_feat[None, :], axis=1)
            nearest_label = valid_labels[distances.argmin()]
            
            noise_indices = np.where(noise_mask)[0]
            new_labels[noise_indices[i]] = nearest_label
        
        n_reassigned = noise_mask.sum()
        LOGGER.info(f"Reassigned {n_reassigned} noise points to nearest clusters")
        
        return new_labels
    
    else:
        raise ValueError(f"Unknown noise handling method: {method}")