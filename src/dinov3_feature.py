"""Lightweight DINOv3 feature extraction for zero-shot prompt generation."""
from __future__ import annotations

from dataclasses import dataclass
import math
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
    objectness_smoothing_kernel: int = 1  # 对象性图平滑核尺寸（奇数，<=1 表示禁用）
    objectness_contrast_gamma: float = 1.0  # 对象性图的伽马调整（<1 提升对比度）

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
        model = model.to(device=self.device, dtype=self.dtype)

        # Attempt to infer patch size from the loaded model for consistency checks
        patch_size = getattr(self.config, "patch_size", None)
        inferred_patch: Optional[int] = None
        if hasattr(model, "patch_embed"):
            embed = getattr(model, "patch_embed")
            if hasattr(embed, "patch_size"):
                raw_size = embed.patch_size
                if isinstance(raw_size, tuple):
                    inferred_patch = int(raw_size[0])
                elif isinstance(raw_size, int):
                    inferred_patch = int(raw_size)
        if inferred_patch and inferred_patch > 0:
            if patch_size and patch_size != inferred_patch:
                LOGGER.warning(
                    "Configured patch size %d differs from model patch size %d; using model value",
                    patch_size,
                    inferred_patch,
                )
            object.__setattr__(self.config, "patch_size", inferred_patch)

        return model
    
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

        # Ensure float32 precision for stable PCA and replace invalid values
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_cpu = features.detach().to(torch.float32).cpu().numpy()

        # Additional safeguard on the numpy array to handle any residual NaNs/Infs that
        # might appear after the device transfer (seen on certain driver/toolkit combos).
        if not np.isfinite(features_cpu).all():
            invalid_mask = ~np.isfinite(features_cpu)
            invalid_rows = np.any(invalid_mask, axis=1)
            num_invalid = int(invalid_rows.sum())
            LOGGER.warning(
                "PCA input contained %d rows with non-finite values; sanitizing via np.nan_to_num",
                num_invalid,
            )
            features_cpu = np.nan_to_num(features_cpu, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(features_cpu).all():
            raise ValueError("Encountered non-finite values after sanitizing features for PCA")

        total_variance = float(np.var(features_cpu, axis=0, dtype=np.float64).sum())
        if not np.isfinite(total_variance) or total_variance <= 1e-7:
            LOGGER.warning(
                "Skipping PCA: insufficient variance detected (total_var=%.3e)", total_variance
            )
            return features

        if not self.pca_fitted:
            from sklearn.decomposition import PCA
            self.pca = PCA(
                n_components=self.config.pca_dim,
                random_state=0,
                whiten=True,
            )
            reduced = self.pca.fit_transform(features_cpu)
            self.pca_fitted = True
            explained_sum = float(np.nansum(self.pca.explained_variance_ratio_))
            if not np.isfinite(explained_sum) or explained_sum <= 0.0:
                LOGGER.warning(
                    "PCA reported invalid explained variance (sum=%s); keeping float32 features",
                    str(explained_sum),
                )
                self.pca_fitted = False
                self.pca = None
                return features
            LOGGER.info(
                f"PCA fitted: {features.shape[1]} -> {self.config.pca_dim} dims, "
                f"explained variance: {explained_sum:.3f}"
            )
        else:
            reduced = self.pca.transform(features_cpu)

        return torch.from_numpy(reduced).to(features.device, dtype=torch.float32)

    def _compute_objectness(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        计算对象性评分

        Args:
            patch_features: [N, D] tensor

        Returns:
            Objectness scores [N] tensor
        """
        # 归一化特征（使用 float32 以获得更稳定的相似度）
        patch_features = patch_features.to(dtype=torch.float32)
        num_tokens = patch_features.shape[0]
        if num_tokens <= 1:
            return torch.zeros(num_tokens, device=patch_features.device, dtype=torch.float32)

        patch_features_norm = F.normalize(patch_features, dim=-1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())

        # 对每个 patch，计算其与最相似的 K 个邻居的平均相似度
        K = min(20, num_tokens - 1)
        topk_sim, _ = torch.topk(similarity_matrix, k=K + 1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)  # 排除自己

        # 对象性 = 1 - 相似度（越独特 = 越可能是物体）
        objectness = 1 - avg_similarity

        return objectness

    @torch.inference_mode()
    def extract_features(self, image: np.ndarray) -> Dict[str, object]:
        """Return patch embeddings, class token features and optional attention."""

        inputs = self._prepare(image)
        processed_height, processed_width = inputs.shape[-2:]
        layers = self._gather_layers(inputs)
        separated = [self._split_tokens(layer) for layer in layers]

        # 提取每层的 patch tokens
        layer_patch_tokens = [pair[0].squeeze(0) for pair in separated]  # List of [P, D_i]
        layer_cls_tokens = [pair[1].squeeze(0) for pair in separated]     # List of [D_i]

        # 多层特征融合
        LOGGER.debug(f"Fusing {len(layer_patch_tokens)} layers with method: {self.config.fusion_method}")
        fused_patch_tokens = self._fuse_multilayer_features(layer_patch_tokens)

        # 使用 float32 以确保聚类稳定，并清除潜在的非有限值
        fused_patch_tokens = fused_patch_tokens.to(dtype=torch.float32)
        fused_patch_tokens = torch.nan_to_num(
            fused_patch_tokens, nan=0.0, posinf=0.0, neginf=0.0
        )
        objectness_tokens = (
            fused_patch_tokens.clone() if self.config.enable_objectness else None
        )
        if not torch.isfinite(fused_patch_tokens).all():
            raise ValueError("Encountered non-finite fused patch tokens before PCA")

        # PCA 降维（可选）
        if self.config.enable_pca:
            fused_patch_tokens = self._apply_pca(fused_patch_tokens)

        fused_patch_tokens = fused_patch_tokens.detach().cpu()

        cls_tokens = (
            torch.stack(layer_cls_tokens, dim=0)
            .mean(dim=0)
            .to(dtype=torch.float32)
            .cpu()
        )

        # 计算 grid size
        num_tokens = fused_patch_tokens.shape[0]
        patch_size = max(1, int(getattr(self.config, "patch_size", 1)))
        tokens_h = processed_height // patch_size
        tokens_w = processed_width // patch_size

        if tokens_h * tokens_w != num_tokens:
            fallback = int(round(math.sqrt(num_tokens)))
            if fallback * fallback == num_tokens:
                LOGGER.warning(
                    "Token grid mismatch (expected %dx%d, got %d tokens); "
                    "falling back to square grid %dx%d",
                    tokens_h,
                    tokens_w,
                    num_tokens,
                    fallback,
                    fallback,
                )
                tokens_h = tokens_w = fallback
            else:
                raise ValueError(
                    f"Patch tokens ({num_tokens}) do not align with patch grid "
                    f"(processed size {processed_height}x{processed_width}, patch {patch_size})"
                )

        # 重塑为 spatial map
        patch_map = fused_patch_tokens.reshape(tokens_h, tokens_w, -1)

        # 获取 attention map
        attention = self._gather_attention(inputs)
        attention_map = None
        if attention is not None:
            attn = attention.mean(dim=1)[0]  # tokens x tokens
            cls_attention = attn[0, 1:]
            attention_map = cls_attention.reshape(tokens_h, tokens_w).cpu().numpy()
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min() + 1e-6
            )

        # 计算对象性（可选）
        objectness_map = None
        if self.config.enable_objectness and objectness_tokens is not None:
            objectness_scores = self._compute_objectness(objectness_tokens)
            objectness_scores = torch.nan_to_num(
                objectness_scores, nan=0.0, posinf=0.0, neginf=0.0
            )
            objectness_scores = torch.clamp(objectness_scores, min=0.0)

            smoothing_kernel = max(1, int(self.config.objectness_smoothing_kernel))
            if smoothing_kernel % 2 == 0:
                smoothing_kernel += 1

            objectness_grid = objectness_scores.view(1, 1, tokens_h, tokens_w)
            if smoothing_kernel > 1:
                pad = smoothing_kernel // 2
                objectness_grid = F.avg_pool2d(
                    objectness_grid,
                    kernel_size=smoothing_kernel,
                    stride=1,
                    padding=pad,
                    count_include_pad=False,
                )

            objectness_tensor = objectness_grid.view(tokens_h, tokens_w)
            objectness_tensor = objectness_tensor - objectness_tensor.min()
            max_val = objectness_tensor.max()
            if max_val > 1e-8:
                objectness_tensor = objectness_tensor / max_val

            gamma = float(getattr(self.config, "objectness_contrast_gamma", 1.0) or 1.0)
            gamma = max(1e-3, gamma)
            if abs(gamma - 1.0) > 1e-3:
                objectness_tensor = objectness_tensor.clamp(min=0.0, max=1.0)
                objectness_tensor = torch.pow(objectness_tensor, gamma)

            flat_scores = objectness_tensor.reshape(-1)
            if flat_scores.numel() >= 16:
                try:
                    lower = torch.quantile(flat_scores, 0.1)
                    upper = torch.quantile(flat_scores, 0.9)
                except RuntimeError:
                    lower = upper = torch.tensor(float("nan"), device=objectness_tensor.device)

                if torch.isfinite(lower) and torch.isfinite(upper) and float(upper - lower) > 1e-6:
                    objectness_tensor = (objectness_tensor - lower) / (upper - lower)
                    objectness_tensor = objectness_tensor.clamp(min=0.0, max=1.0)
                else:
                    mean = flat_scores.mean()
                    std = flat_scores.std()
                    if torch.isfinite(mean) and torch.isfinite(std) and float(std) > 1e-6:
                        normalized = (objectness_tensor - mean) / (std * 2.0)
                        objectness_tensor = torch.sigmoid(torch.clamp(normalized, -4.0, 4.0))

            objectness_map = objectness_tensor.cpu().numpy()

        patch_tokens_np = fused_patch_tokens.numpy()
        patch_map_np = patch_map.numpy()
        cls_token_np = cls_tokens.numpy()

        patch_tokens_np = fused_patch_tokens.numpy()
        patch_map_np = patch_map.numpy()
        cls_token_np = cls_tokens.numpy()

        return {
            "patch_tokens": patch_tokens_np,
            "cls_token": cls_token_np,
            "grid_size": (tokens_h, tokens_w),
            "patch_map": patch_map_np,
            "attention_map": attention_map,
            "objectness_map": objectness_map,  # 新增
            "processed_image_shape": (processed_height, processed_width),
        }

    def to(self, device: torch.device | str) -> "DINOv3FeatureExtractor":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self