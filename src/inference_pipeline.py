"""Zero-shot segmentation pipeline combining DINOv3 proposals with SAM2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .prompt_generator import (
    ClusterConfig,
    PromptConfig,
    RegionProposal,
    kmeans_cluster,
    labels_to_regions,
    expand_region_instances,
    proposals_to_prompts,
)
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .mask_postprocess import nms_with_objectness, filter_by_combined_score
from .superpixel_helper import SuperpixelGenerator, SuperpixelConfig
from .graph_clustering import GraphClusterer, GraphClusterConfig, merge_small_clusters
from .density_clustering import DensityClusterer, DensityClusterConfig, handle_noise_points
from .crf_refinement import CRFRefiner, CRFConfig, bilateral_filter_mask
from .utils import LOGGER


@dataclass
class PipelineConfig:
    """High level knobs for the zero-shot segmentation routine."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    
    # 新增：高级聚类选项
    use_superpixels: bool = False
    superpixel: SuperpixelConfig = field(default_factory=SuperpixelConfig)
    
    use_graph_clustering: bool = False
    graph_cluster: GraphClusterConfig = field(default_factory=GraphClusterConfig)
    
    use_density_clustering: bool = False
    density_cluster: DensityClusterConfig = field(default_factory=DensityClusterConfig)
    
    # CRF 细化
    crf: CRFConfig = field(default_factory=CRFConfig)


@dataclass
class PipelineResult:
    """Rich return type that exposes intermediate artefacts for inspection."""

    masks: List[np.ndarray]
    proposals: List[RegionProposal]
    label_map: np.ndarray
    attention_map: Optional[np.ndarray]
    objectness_map: Optional[np.ndarray]
    prompts: Dict[str, List]
    cluster_centroids: Optional[np.ndarray]
    superpixel_labels: Optional[np.ndarray] = None  # 新增


class ZeroShotSegmentationPipeline:
    """Orchestrates DINOv3 feature extraction, clustering and SAM2 inference."""

    def __init__(
        self,
        dinov3_config: Dinov3Config,
        sam2_config: Sam2Config,
        pipeline_config: PipelineConfig,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.float32,
        extractor: Optional[DINOv3FeatureExtractor] = None,
        segmenter: Optional[SAM2Segmenter] = None,
    ) -> None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.extractor = extractor or DINOv3FeatureExtractor(dinov3_config, device, torch_dtype)
        self.segmenter = segmenter or SAM2Segmenter(sam2_config, device, torch_dtype)
        self.config = pipeline_config
        
        # 初始化可选组件
        self.superpixel_gen = None
        if self.config.use_superpixels:
            self.superpixel_gen = SuperpixelGenerator(self.config.superpixel)
        
        self.graph_clusterer = None
        if self.config.use_graph_clustering:
            self.graph_clusterer = GraphClusterer(self.config.graph_cluster)
        
        self.density_clusterer = None
        if self.config.use_density_clustering:
            self.density_clusterer = DensityClusterer(self.config.density_cluster)
        
        self.crf_refiner = CRFRefiner(self.config.crf)

    def _cluster_basic(self, patch_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基础 KMeans 聚类"""
        grid_h, grid_w, _ = patch_map.shape
        features = patch_map.reshape(-1, patch_map.shape[-1])
        labels, centroids = kmeans_cluster(features, self.config.cluster)
        label_map = labels.reshape(grid_h, grid_w)
        return label_map, centroids
    
    def _cluster_with_superpixels(
        self,
        image: np.ndarray,
        patch_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用超像素辅助的聚类"""
        # 生成超像素
        superpixel_labels = self.superpixel_gen.generate_superpixels(image)
        
        # 将 patch 特征聚合到超像素
        patch_features = patch_map.reshape(-1, patch_map.shape[-1])
        superpixel_features, superpixel_ids = self.superpixel_gen.aggregate_features_by_superpixels(
            patch_features,
            superpixel_labels,
            image.shape[:2]
        )
        
        # 对超像素特征聚类
        if self.config.use_graph_clustering:
            # 构建超像素邻接图
            edges, edge_weights = self.superpixel_gen.create_adjacency_graph(superpixel_labels)
            
            # 图聚类
            cluster_labels = self.graph_clusterer.cluster(
                superpixel_features,
                edges,
                edge_weights
            )
        elif self.config.use_density_clustering:
            # 密度聚类
            cluster_labels = self.density_clusterer.cluster(superpixel_features)
            cluster_labels = handle_noise_points(cluster_labels, superpixel_features)
        else:
            # 普通 KMeans
            from .prompt_generator import kmeans_cluster
            cluster_labels, centroids = kmeans_cluster(
                superpixel_features,
                self.config.cluster
            )
        
        # 将聚类标签映射回像素级
        height, width = image.shape[:2]
        pixel_labels = np.zeros((height, width), dtype=np.int32)
        
        for sp_id, cluster_id in zip(superpixel_ids, cluster_labels):
            pixel_labels[superpixel_labels == sp_id] = cluster_id
        
        # 下采样到 patch grid 用于后续处理
        grid_h, grid_w, _ = patch_map.shape
        from PIL import Image
        label_pil = Image.fromarray(pixel_labels.astype(np.int32), mode="I")
        label_map = np.array(
            label_pil.resize((grid_w, grid_h), resample=Image.NEAREST), dtype=np.int32
        )

        return label_map, superpixel_features, superpixel_labels

    def _apply_nms(
        self, 
        masks: List[np.ndarray],
        proposals: List[RegionProposal],
        nms_config: Dict
    ) -> Tuple[List[np.ndarray], List[RegionProposal]]:
        """应用 NMS 去重"""
        if not masks or not nms_config.get("enable_nms", False):
            return masks, proposals
        
        objectness_scores = [p.objectness for p in proposals]
        
        keep_masks, keep_indices = nms_with_objectness(
            masks,
            objectness_scores,
            sam2_confidence=None,
            iou_threshold=nms_config.get("iou_threshold", 0.6),
            objectness_weight=nms_config.get("objectness_weight", 0.5),
            confidence_weight=nms_config.get("confidence_weight", 0.3),
            area_weight=nms_config.get("area_weight", 0.2)
        )
        
        keep_proposals = [proposals[i] for i in keep_indices]
        
        LOGGER.info(f"NMS: {len(masks)} masks -> {len(keep_masks)} masks")
        
        return keep_masks, keep_proposals
    
    def _refine_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """细化掩码边界"""
        if self.config.crf.enable:
            # CRF 细化
            refined = self.crf_refiner.refine_masks_batch(image, masks)
            LOGGER.info("Applied CRF refinement")
        else:
            # 简单的双边滤波
            refined = []
            for mask in masks:
                refined_mask = bilateral_filter_mask(mask, image)
                refined.append(refined_mask)
        
        return refined

    def run(
        self, 
        image: np.ndarray, 
        nms_config: Optional[Dict] = None
    ) -> PipelineResult:
        """
        运行完整的分割 pipeline
        
        Args:
            image: RGB 图像 [H, W, 3]
            nms_config: NMS 配置（可选）
            
        Returns:
            PipelineResult 包含掩码和中间结果
        """
        # 提取多层融合特征
        LOGGER.info("Extracting DINOv3 features...")
        feats = self.extractor.extract_features(image)
        patch_map = feats["patch_map"]
        if hasattr(patch_map, "detach"):
            patch_map = patch_map.detach().cpu().numpy()
        
        objectness_map = feats.get("objectness_map")
        
        # 聚类
        LOGGER.info("Clustering features...")
        superpixel_labels = None
        
        if self.config.use_superpixels:
            label_map, centroids, superpixel_labels = self._cluster_with_superpixels(
                image, patch_map
            )
        else:
            label_map, centroids = self._cluster_basic(patch_map)
        
        # 生成候选区域
        LOGGER.info("Generating region proposals...")
        semantic_proposals = labels_to_regions(
            label_map,
            image.shape[:2],
            self.config.cluster,
            objectness_map=objectness_map
        )

        if not semantic_proposals:
            LOGGER.warning("No proposals survived filtering; returning empty result")
            return PipelineResult(
                [],
                [],
                label_map,
                feats.get("attention_map"),
                objectness_map,
                {"boxes": [], "points": [], "labels": []},
                centroids,
                superpixel_labels
            )

        proposals = expand_region_instances(
            semantic_proposals,
            self.config.prompt,
            self.config.cluster,
            patch_map,
            image.shape[:2],
        )

        if not proposals:
            LOGGER.warning("Instance expansion returned no proposals; falling back to semantic regions")
            proposals = semantic_proposals

        # 生成 SAM2 prompts
        LOGGER.info(f"Converting {len(proposals)} proposals to prompts...")
        boxes, points, labels = proposals_to_prompts(
            proposals,
            self.config.prompt,
            patch_map=patch_map,
            image_shape=image.shape[:2],
            cluster_config=self.config.cluster,
        )

        # SAM2 分割
        LOGGER.info("Running SAM2 segmentation...")
        masks = self.segmenter.segment_batched(image, boxes, points=points, labels=labels)

        for proposal, mask in zip(proposals, masks):
            binary_mask = mask.astype(np.uint8)
            proposal.mask = binary_mask
            ys, xs = np.nonzero(binary_mask)
            if len(xs) == 0:
                continue
            proposal.bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            proposal.centroid = (int(np.round(xs.mean())), int(np.round(ys.mean())))

        # NMS 去重
        if nms_config is None:
            nms_config = {}
        
        masks, proposals = self._apply_nms(masks, proposals, nms_config)
        
        # CRF 细化（可选）
        if self.config.crf.enable and masks:
            LOGGER.info("Refining mask boundaries...")
            masks = self._refine_masks(image, masks)
        
        # 更新 prompts
        boxes, points, labels = proposals_to_prompts(
            proposals,
            self.config.prompt,
            patch_map=patch_map,
            image_shape=image.shape[:2],
            cluster_config=self.config.cluster,
        )
        
        prompts: Dict[str, List] = {"boxes": boxes, "points": points, "labels": labels}
        
        LOGGER.info(f"Pipeline complete: {len(masks)} final masks")
        
        return PipelineResult(
            masks, 
            proposals, 
            label_map, 
            feats.get("attention_map"), 
            objectness_map,
            prompts, 
            centroids,
            superpixel_labels
        )


def build_pipeline(
    dinov3_cfg: Dinov3Config,
    sam2_cfg: Sam2Config,
    pipeline_cfg: Optional[PipelineConfig] = None,
    device: str = "cuda",
    dtype: str = "float32",
    extractor: Optional[DINOv3FeatureExtractor] = None,
    segmenter: Optional[SAM2Segmenter] = None,
) -> ZeroShotSegmentationPipeline:
    pipeline_cfg = pipeline_cfg or PipelineConfig()
    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )