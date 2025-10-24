"""Graph-based clustering for superpixel features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .utils import LOGGER


@dataclass
class GraphClusterConfig:
    """Configuration for graph-based clustering."""
    
    method: str = "spectral"  # spectral, louvain, label_propagation
    n_clusters: int = 6
    affinity: str = "rbf"  # rbf, cosine, nearest_neighbors
    gamma: float = 1.0  # RBF kernel parameter
    n_neighbors: int = 10  # KNN 邻居数
    
    # Louvain 参数
    resolution: float = 1.0
    
    # Label Propagation 参数
    max_iter: int = 30


class GraphClusterer:
    """Graph-based clustering for instance separation."""
    
    def __init__(self, config: GraphClusterConfig):
        self.config = config
    
    def cluster(
        self,
        features: np.ndarray,
        adjacency_edges: Optional[np.ndarray] = None,
        adjacency_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        执行图聚类
        
        Args:
            features: [N, D] 特征矩阵
            adjacency_edges: [E, 2] 邻接边（可选）
            adjacency_weights: [E] 边权重（可选）
            
        Returns:
            labels: [N] 聚类标签
        """
        if self.config.method == "spectral":
            return self._spectral_clustering(features, adjacency_edges, adjacency_weights)
        elif self.config.method == "louvain":
            return self._louvain_clustering(features, adjacency_edges, adjacency_weights)
        elif self.config.method == "label_propagation":
            return self._label_propagation(features, adjacency_edges, adjacency_weights)
        else:
            raise ValueError(f"Unknown clustering method: {self.config.method}")
    
    def _build_affinity_matrix(
        self,
        features: np.ndarray,
        adjacency_edges: Optional[np.ndarray] = None,
        adjacency_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        构建亲和度矩阵
        
        Returns:
            affinity: [N, N] 亲和度矩阵
        """
        n = len(features)
        
        # 计算特征相似度
        if self.config.affinity == "rbf":
            # RBF kernel
            from sklearn.metrics.pairwise import rbf_kernel
            affinity = rbf_kernel(features, gamma=self.config.gamma)
            
        elif self.config.affinity == "cosine":
            # 余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            affinity = cosine_similarity(features)
            affinity = (affinity + 1) / 2  # 归一化到 [0, 1]
            
        elif self.config.affinity == "nearest_neighbors":
            # KNN 图
            from sklearn.neighbors import kneighbors_graph
            knn_graph = kneighbors_graph(
                features,
                n_neighbors=self.config.n_neighbors,
                mode='connectivity',
                include_self=False
            )
            affinity = 0.5 * (knn_graph + knn_graph.T)
            affinity = affinity.toarray()
            
        else:
            raise ValueError(f"Unknown affinity: {self.config.affinity}")
        
        # 如果提供了邻接信息，融合空间邻接约束
        if adjacency_edges is not None and adjacency_weights is not None:
            # 创建邻接矩阵
            adj_matrix = np.zeros((n, n), dtype=np.float32)
            for (i, j), weight in zip(adjacency_edges, adjacency_weights):
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight
            
            # 融合：特征相似度 * 空间邻接
            # 只有空间上相邻的节点才保留边
            affinity = affinity * (adj_matrix > 0)
        
        return affinity
    
    def _spectral_clustering(
        self,
        features: np.ndarray,
        adjacency_edges: Optional[np.ndarray],
        adjacency_weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """谱聚类"""
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        # 构建亲和度矩阵
        affinity = self._build_affinity_matrix(features, adjacency_edges, adjacency_weights)
        
        # 谱聚类
        clustering = SpectralClustering(
            n_clusters=self.config.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )
        
        labels = clustering.fit_predict(affinity)
        
        LOGGER.info(f"Spectral clustering: {len(features)} nodes -> {len(np.unique(labels))} clusters")
        
        return labels
    
    def _louvain_clustering(
        self,
        features: np.ndarray,
        adjacency_edges: Optional[np.ndarray],
        adjacency_weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """Louvain 社区发现"""
        try:
            import networkx as nx
            from networkx.algorithms import community
        except ImportError:
            raise ImportError("Please install networkx: pip install networkx")
        
        # 构建亲和度矩阵
        affinity = self._build_affinity_matrix(features, adjacency_edges, adjacency_weights)
        
        # 创建图
        G = nx.Graph()
        n = len(features)
        
        # 添加边（过滤小权重边）
        threshold = np.percentile(affinity[affinity > 0], 50) if (affinity > 0).any() else 0
        for i in range(n):
            for j in range(i + 1, n):
                if affinity[i, j] > threshold:
                    G.add_edge(i, j, weight=affinity[i, j])
        
        # Louvain 聚类
        communities = community.louvain_communities(
            G,
            weight='weight',
            resolution=self.config.resolution,
            seed=0
        )
        
        # 转换为标签数组
        labels = np.zeros(n, dtype=np.int32)
        for cluster_id, nodes in enumerate(communities):
            for node in nodes:
                labels[node] = cluster_id
        
        LOGGER.info(f"Louvain clustering: {len(features)} nodes -> {len(communities)} communities")
        
        return labels
    
    def _label_propagation(
        self,
        features: np.ndarray,
        adjacency_edges: Optional[np.ndarray],
        adjacency_weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """标签传播聚类"""
        try:
            from sklearn.semi_supervised import LabelPropagation
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        # 构建亲和度矩阵
        affinity = self._build_affinity_matrix(features, adjacency_edges, adjacency_weights)
        
        # 初始化：随机选择一些种子点
        n = len(features)
        n_seeds = min(self.config.n_clusters * 2, n // 10)
        seed_indices = np.random.choice(n, size=n_seeds, replace=False)
        
        # 初始标签：-1 表示未标记
        initial_labels = np.full(n, -1, dtype=np.int32)
        for i, idx in enumerate(seed_indices):
            initial_labels[idx] = i % self.config.n_clusters
        
        # 标签传播
        lp = LabelPropagation(
            kernel='rbf',
            gamma=self.config.gamma,
            max_iter=self.config.max_iter
        )
        
        # 注意：LabelPropagation 需要至少有一些标记样本
        # 我们使用种子点作为标记样本
        labels = lp.fit(affinity, initial_labels).transitive_closure_
        
        LOGGER.info(f"Label propagation: {len(features)} nodes -> {len(np.unique(labels))} clusters")
        
        return labels


def merge_small_clusters(
    labels: np.ndarray,
    features: np.ndarray,
    min_cluster_size: int = 50
) -> np.ndarray:
    """
    合并小聚类到最近的大聚类
    
    Args:
        labels: 聚类标签
        features: 特征向量
        min_cluster_size: 最小聚类大小
        
    Returns:
        合并后的标签
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 找出小聚类
    small_clusters = unique_labels[counts < min_cluster_size]
    large_clusters = unique_labels[counts >= min_cluster_size]
    
    if len(small_clusters) == 0:
        return labels
    
    # 计算大聚类的质心
    centroids = []
    for label in large_clusters:
        mask = labels == label
        centroid = features[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # 合并小聚类
    new_labels = labels.copy()
    for small_label in small_clusters:
        mask = labels == small_label
        small_features = features[mask]
        small_centroid = small_features.mean(axis=0)
        
        # 找最近的大聚类
        distances = np.linalg.norm(centroids - small_centroid[None, :], axis=1)
        nearest_large_label = large_clusters[distances.argmin()]
        
        new_labels[mask] = nearest_large_label
    
    LOGGER.info(f"Merged {len(small_clusters)} small clusters")
    
    return new_labels