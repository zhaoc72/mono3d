"""Lightweight DINOv3 feature extraction for zero-shot prompt generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .utils import LOGGER


@dataclass
class Dinov3Config:
    """Configuration describing how to load and query a DINOv3 backbone."""

    repo_or_dir: str = "facebookresearch/dinov2"
    model_name: str = "dinov2_vitl14"
    use_torch_hub: bool = True
    checkpoint_path: Optional[str] = None
    image_size: int = 518
    output_layers: Sequence[int] = (4, 8, 12)  # 多层特征
    patch_size: int = 14
    normalize: bool = True
    torchhub_source: Optional[str] = None
    output_layer: Optional[int] = None
    
    # 新增：多层特征融合配置
    layer_weights: Optional[Sequence[float]] = None  # 层权重，如 (0.3, 0.3, 0.4)
    fusion_method: str = "weighted_concat"  # weighted_concat, weighted_sum, concat
    enable_pca: bool = False  # 是否启用 PCA 降维
    pca_dim: int = 32  # PCA 降维目标维度
    enable_objectness: bool = True  # 是否计算对象性评分

    def __post_init__(self) -> None:
        """Normalize legacy configuration options."""

        layers: Sequence[int]
        if isinstance(self.output_layers, int):
            layers = (self.output_layers,)
        elif isinstance(self.output_layers, (list, tuple)):
            layers = tuple(self.output_layers)
        else:
            layers = tuple(self.output_layers)

        if self.output_layer is not None:
            layers = (self.output_layer,)

        normalized: list[int] = []
        for index, layer in enumerate(layers):
            if layer < 0:
                layer = -layer - 1
            if layer < 0:
                raise ValueError(
                    f"Invalid output layer specification at position {index}: {layers}"
                )
            normalized.append(int(layer))

        object.__setattr__(self, "output_layers", tuple(normalized))
        object.__setattr__(self, "output_layer", None)
        
        # 初始化层权重
        if self.layer_weights is None:
            n_layers = len(self.output_layers)
            # 默认：后面的层权重更大
            weights = [0.2 + 0.3 * (i / max(1, n_layers - 1)) for i in range(n_layers)]
            weights = [w / sum(weights) for w in weights]  # 归一化
            object.__setattr__(self, "layer_weights", tuple(weights))
        else:
            # 验证权重
            if len(self.layer_weights) != len(self.output_layers):
                raise ValueError(
                    f"layer_weights length {len(self.layer_weights)} must match "
                    f"output_layers length {len(self.output_layers)}"
                )
            # 归一化权重
            weights = list(self.layer_weights)
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            object.__setattr__(self, "layer_weights", tuple(weights))


class DINOv3FeatureExtractor:
    """Wrapper that exposes patch embeddings and attention maps from DINOv3."""

    def __init__(self, config: Dinov3Config, device: torch.device | str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = self._load_model()
        self.model.eval()
        
        # PCA 组件（延迟初始化）
        self.pca = None
        self.pca_fitted = False
        
        transform_steps = [
            transforms.ToPILImage(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ]
        if config.normalize:
            transform_steps.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = transforms.Compose(transform_steps)
    
    def _load_model(self) -> torch.nn.Module:
        if self.config.use_torch_hub:
            LOGGER.info(
                "Loading DINOv3 weights %s from %s", self.config.model_name, self.config.repo_or_dir
            )
            source = self.config.torchhub_source
            if source is None:
                source = "local" if self._is_local_repo(self.config.repo_or_dir) else "github"
            kwargs = {
                "trust_repo": True,
                "source": source,
                "pretrained": self.config.checkpoint_path is None,
            }
            model = torch.hub.load(self.config.repo_or_dir, self.config.model_name, **kwargs)
        else:
            module = __import__(self.config.repo_or_dir, fromlist=[self.config.model_name])
            model = getattr(module, self.config.model_name)
        if self.config.checkpoint_path:
            state = torch.load(self.config.checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                LOGGER.warning("Missing parameters when loading checkpoint: %s", missing[:5])
            if unexpected:
                LOGGER.warning("Unexpected parameters when loading checkpoint: %s", unexpected[:5])
        return model.to(device=self.device, dtype=self.dtype)
    
    @staticmethod
    def _is_local_repo(repo_or_dir: str) -> bool:
        return Path(repo_or_dir).exists()

    def _prepare(self, image: np.ndarray) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)
    
    def _gather_layers(self, inputs: torch.Tensor) -> Sequence[torch.Tensor | Tuple]:
        if not hasattr(self.model, "get_intermediate_layers"):
            raise RuntimeError("DINOv3 model must expose get_intermediate_layers")
        max_offset = max(self.config.output_layers)
        raw_layers = self.model.get_intermediate_layers(
            inputs,
            n=max_offset + 1,
            reshape=False,
            return_class_token=True,
        )
        selected = [raw_layers[-(offset + 1)] for offset in self.config.output_layers]
        return selected

    @staticmethod
    def _split_tokens(layer: torch.Tensor | Sequence[object]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (patch_tokens, cls_token) from various DINOv3 layer outputs."""

        if isinstance(layer, torch.Tensor):
            if layer.ndim != 3:
                raise TypeError(
                    f"Unsupported tensor shape for intermediate layer: {tuple(layer.shape)}"
                )
            return layer[:, 1:, :], layer[:, 0, :]

        if isinstance(layer, (tuple, list)):
            patch_candidate: Optional[torch.Tensor] = None
            cls_candidate: Optional[torch.Tensor] = None

            for item in layer:
                if isinstance(item, torch.Tensor):
                    if item.ndim == 3:
                        if item.shape[1] == 1:
                            cls_candidate = item[:, 0, :]
                        elif item.shape[1] > 1:
                            patch_candidate = item
                    elif item.ndim == 2:
                        cls_candidate = item
                elif isinstance(item, (tuple, list)):
                    try:
                        nested_patch, nested_cls = DINOv3FeatureExtractor._split_tokens(item)
                    except TypeError:
                        continue
                    else:
                        patch_candidate = patch_candidate or nested_patch
                        cls_candidate = cls_candidate or nested_cls

            if patch_candidate is None:
                # Some implementations return a single tensor that still includes the class token.
                for item in layer:
                    if isinstance(item, torch.Tensor) and item.ndim == 3:
                        patch_candidate = item[:, 1:, :]
                        cls_candidate = item[:, 0, :]
                        break

            if patch_candidate is None:
                raise TypeError("Unable to locate patch tokens in intermediate layer output")

            if cls_candidate is None:
                # Fall back to deriving the class token from the first patch position if available.
                if patch_candidate.ndim != 3 or patch_candidate.shape[1] == 0:
                    raise TypeError("Intermediate layer output is missing a class token")
                cls_candidate = patch_candidate[:, 0, :]
                patch_candidate = patch_candidate[:, 1:, :]

            return patch_candidate, cls_candidate

        raise TypeError(f"Unsupported intermediate layer output type: {type(layer)!r}")

    def _gather_attention(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        if not hasattr(self.model, "get_last_selfattention"):
            return None
        try:
            attention = self.model.get_last_selfattention(inputs)
        except TypeError:
            attention = self.model.get_last_selfattention()
        return attention.detach().to("cpu") if isinstance(attention, torch.Tensor) else None

    def _fuse_multilayer_features(
        self, 
        layer_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        融合多层特征
        
        Args:
            layer_features: List of [N, D_i] tensors
            
        Returns:
            Fused features [N, D_out]
        """
        if len(layer_features) == 1:
            return layer_features[0]
        
        # L2 归一化每一层
        normalized = [F.normalize(feat, dim=-1) for feat in layer_features]
        
        if self.config.fusion_method == "weighted_concat":
            # 加权后拼接
            weighted = [feat * w for feat, w in zip(normalized, self.config.layer_weights)]
            fused = torch.cat(weighted, dim=-1)
            
        elif self.config.fusion_method == "weighted_sum":
            # 加权求和（要求所有层维度相同）
            weighted = [feat * w for feat, w in zip(normalized, self.config.layer_weights)]
            fused = torch.stack(weighted, dim=0).sum(dim=0)
            
        elif self.config.fusion_method == "concat":
            # 简单拼接
            fused = torch.cat(normalized, dim=-1)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
        
        # 最终 L2 归一化
        fused = F.normalize(fused, dim=-1)
        
        return fused

    def _apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """
        对特征应用 PCA 降维
        
        Args:
            features: [N, D] tensor
            
        Returns:
            Reduced features [N, pca_dim]
        """
        if not self.config.enable_pca:
            return features
        
        # 转到 CPU 进行 PCA
        features_cpu = features.cpu().numpy()
        
        if not self.pca_fitted:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.config.pca_dim, random_state=0)
            reduced = self.pca.fit_transform(features_cpu)
            self.pca_fitted = True
            LOGGER.info(
                f"PCA fitted: {features.shape[1]} -> {self.config.pca_dim} dims, "
                f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}"
            )
        else:
            reduced = self.pca.transform(features_cpu)
        
        return torch.from_numpy(reduced).to(features.device, dtype=features.dtype)

    def _compute_objectness(self, patch_features: torch.Tensor) -> np.ndarray:
        """
        计算对象性评分
        
        Args:
            patch_features: [N, D] tensor
            
        Returns:
            Objectness scores [N] numpy array
        """
        # 归一化特征
        patch_features_norm = F.normalize(patch_features, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())
        
        # 对每个 patch，计算其与最相似的 K 个邻居的平均相似度
        K = min(20, len(patch_features) - 1)
        topk_sim, _ = torch.topk(similarity_matrix, k=K + 1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)  # 排除自己
        
        # 对象性 = 1 - 相似度（越独特 = 越可能是物体）
        objectness = (1 - avg_similarity).cpu().numpy()
        
        return objectness

    @torch.inference_mode()
    def extract_features(self, image: np.ndarray) -> Dict[str, object]:
        """Return patch embeddings, class token features and optional attention."""

        inputs = self._prepare(image)
        layers = self._gather_layers(inputs)
        separated = [self._split_tokens(layer) for layer in layers]
        
        # 提取每层的 patch tokens
        layer_patch_tokens = [pair[0].squeeze(0) for pair in separated]  # List of [P, D_i]
        layer_cls_tokens = [pair[1].squeeze(0) for pair in separated]     # List of [D_i]
        
        # 多层特征融合
        LOGGER.debug(f"Fusing {len(layer_patch_tokens)} layers with method: {self.config.fusion_method}")
        fused_patch_tokens = self._fuse_multilayer_features(layer_patch_tokens)
        
        # PCA 降维（可选）
        if self.config.enable_pca:
            fused_patch_tokens = self._apply_pca(fused_patch_tokens)
        
        # 转到 CPU
        fused_patch_tokens = fused_patch_tokens.to("cpu")
        cls_tokens = torch.stack(layer_cls_tokens, dim=0).mean(dim=0).to("cpu")
        
        # 计算 grid size
        num_tokens = fused_patch_tokens.shape[0]
        grid_size = int(round(np.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"Number of patch tokens ({num_tokens}) does not form a square grid"
            )
        
        # 重塑为 spatial map
        patch_map = fused_patch_tokens.reshape(grid_size, grid_size, -1)
        
        # 获取 attention map
        attention = self._gather_attention(inputs)
        attention_map = None
        if attention is not None:
            attn = attention.mean(dim=1)[0]  # tokens x tokens
            cls_attention = attn[0, 1:]
            attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min() + 1e-6
            )
        
        # 计算对象性（可选）
        objectness_map = None
        if self.config.enable_objectness:
            objectness_scores = self._compute_objectness(fused_patch_tokens)
            objectness_map = objectness_scores.reshape(grid_size, grid_size)
            # 归一化
            objectness_map = (objectness_map - objectness_map.min()) / (
                objectness_map.max() - objectness_map.min() + 1e-8
            )

        return {
            "patch_tokens": fused_patch_tokens,
            "cls_token": cls_tokens,
            "grid_size": (grid_size, grid_size),
            "patch_map": patch_map,
            "attention_map": attention_map,
            "objectness_map": objectness_map,  # 新增
        }

    def to(self, device: torch.device | str) -> "DINOv3FeatureExtractor":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self