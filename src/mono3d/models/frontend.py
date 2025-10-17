"""封装视觉前端模型组件。

本模块聚合了 DINOv3、SAM2 和 Depth Anything V2 等基础模型。
在离线环境下我们会优先从 ``checkpoints/pretrained`` 目录加载
用户提供的检查点文件，并在缺失依赖或权重时提供退化实现，
以保证单元测试和最小化 demo 能够正常运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Iterable

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_SEARCH_PATHS: List[Path] = [
    PROJECT_ROOT / "checkpoints" / "pretrained",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT,
]


def _resolve_checkpoint(explicit: Optional[str], candidates: Iterable[str]) -> Optional[Path]:
    """Resolve a checkpoint path from explicit or candidate filenames.

    Args:
        explicit: User-specified path (absolute or relative).
        candidates: Candidate filenames to probe under the default search paths.

    Returns:
        A resolved :class:`Path` if the file exists, otherwise ``None``.
    """

    def _probe(path: Path) -> Optional[Path]:
        if path.is_file():
            return path
        return None

    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_absolute():
            for base in CHECKPOINT_SEARCH_PATHS:
                resolved = _probe(base / explicit_path)
                if resolved:
                    return resolved
        else:
            resolved = _probe(explicit_path)
            if resolved:
                return resolved

    for candidate in candidates:
        for base in CHECKPOINT_SEARCH_PATHS:
            resolved = _probe(base / candidate)
            if resolved:
                log.debug("Resolved checkpoint %s", resolved)
                return resolved

    return None


@dataclass(frozen=True)
class _DINOv3Config:
    alias: str
    checkpoint_names: List[str]


_DINOV3_VARIANTS: Dict[str, _DINOv3Config] = {
    "vits16": _DINOv3Config(
        alias="vit_small_patch16_224",
        checkpoint_names=["dinov3_vits16_pretrain_lvd1689m-08c60483.pth"],
    ),
    "vits16plus": _DINOv3Config(
        alias="vit_small_patch16_224",
        checkpoint_names=["dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"],
    ),
    "vitb14": _DINOv3Config(
        alias="vit_base_patch14_224",
        checkpoint_names=[
            "dinov3_vitb14_pretrain.pth",
            "dinov2_vitb14_pretrain.pth",
        ],
    ),
    "default": _DINOv3Config(
        alias="vit_base_patch16_224",
        checkpoint_names=[
            "dinov3_vitb16_pretrain.pth",
            "dinov2_vitb14_pretrain.pth",
        ],
    ),
}


class DINOv3(nn.Module):
    """DINOv3特征提取器
    
    论文: https://arxiv.org/abs/2304.07193
    """
    
    def __init__(
        self,
        backbone: str = 'vit_base_patch16_224',
        weights: Optional[str] = None,
        frozen: bool = True,
        output_layers: Optional[List[int]] = None,
        **kwargs
    ):
        """初始化DINOv3
        
        Args:
            backbone: 骨干网络名称
            weights: 预训练权重路径或URL
            frozen: 是否冻结参数
            output_layers: 要输出的层索引（多尺度特征）
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.frozen = frozen
        self.output_layers = output_layers or [3, 7, 11]

        # 加载模型
        self._load_model(weights)
        
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        
        log.info(f"Initialized DINOv3 ({backbone}), frozen={frozen}")
    
    def _load_model(self, weights: Optional[str]):
        """加载DINOv3模型"""
        resolved_variant = _DINOV3_VARIANTS.get(
            self.backbone_name.lower(), _DINOV3_VARIANTS["default"]
        )

        checkpoint = _resolve_checkpoint(weights, resolved_variant.checkpoint_names)

        try:
            # 优先尝试使用 timm 创建 ViT 结构
            try:
                import timm

                self.model = timm.create_model(
                    resolved_variant.alias,
                    pretrained=False,
                    features_only=True,
                    out_indices=self.output_layers,
                )
            except Exception:
                # 回退到 torch hub (可能需要网络)
                self.model = torch.hub.load(
                    'facebookresearch/dinov2',
                    resolved_variant.alias,
                    pretrained=False,
                )

            if checkpoint:
                state_dict = torch.load(checkpoint, map_location='cpu')
                missing = self.model.load_state_dict(state_dict, strict=False)
                if isinstance(missing, tuple):
                    missing = missing[0] or missing[1]
                log.info(
                    "Loaded DINOv3 weights from %s (strict=%s)",
                    checkpoint,
                    not bool(missing),
                )
            elif weights:
                log.warning("Specified DINOv3 weights not found: %s", weights)

        except Exception as e:
            log.error("Failed to initialise DINOv3 backbone: %s", e)
            log.warning("Using fallback feature extractor instead of ViT")
            self.model = self._create_fallback_model()

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Resize input tensor to match backbone expectations if required."""

        target_hw: Optional[List[int]] = None

        default_cfg = getattr(self.model, 'default_cfg', None)
        if isinstance(default_cfg, dict):
            input_size = default_cfg.get('input_size')
            if input_size and len(input_size) >= 2:
                target_hw = [int(input_size[-2]), int(input_size[-1])]

        if target_hw is None and hasattr(self.model, 'img_size'):
            img_size = getattr(self.model, 'img_size')
            if isinstance(img_size, (list, tuple)):
                if len(img_size) == 2:
                    target_hw = [int(img_size[0]), int(img_size[1])]
                elif len(img_size) == 1:
                    target_hw = [int(img_size[0]), int(img_size[0])]
            elif isinstance(img_size, int):
                target_hw = [img_size, img_size]

        if target_hw and (x.shape[2], x.shape[3]) != tuple(target_hw):
            return F.interpolate(x, size=tuple(target_hw), mode='bilinear', align_corners=False)

        return x

    def _create_fallback_model(self):
        """创建回退模型（简化的CNN）"""
        import torchvision.models as models

        # 使用随机初始化的ResNet，避免在无网络环境下下载权重
        try:
            resnet = models.resnet50(weights=None)
        except TypeError:  # 兼容旧版torchvision API
            resnet = models.resnet50(pretrained=False)

        # 移除最后的全连接层
        return nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            特征字典，包含多尺度特征
        """
        if self.frozen:
            with torch.no_grad():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """实际的前向传播实现"""
        x_prepared = self._prepare_input(x)
        features = {}

        # 如果是DINOv2模型
        if hasattr(self.model, 'get_intermediate_layers'):
            # 获取多层特征
            intermediate = self.model.get_intermediate_layers(
                x_prepared,
                n=self.output_layers,
                return_class_token=True
            )

            for i, (layer_idx, feat) in enumerate(zip(self.output_layers, intermediate)):
                features[f'layer_{layer_idx}'] = feat

            # 最后一层作为全局特征
            features['global'] = intermediate[-1]

        elif isinstance(self.model, nn.Module):
            output = self.model(x_prepared)
            if isinstance(output, (list, tuple)):
                for idx, feat in zip(self.output_layers, output):
                    features[f'layer_{idx}'] = feat
                features['global'] = output[-1]
            else:
                features['global'] = output

        else:
            # 回退模型
            feat = self.model(x_prepared)
            features['global'] = feat

        return features
    
    def extract_patch_features(
        self,
        x: torch.Tensor,
        layer: int = 11
    ) -> torch.Tensor:
        """提取patch级别的特征
        
        Args:
            x: 输入图像 (B, 3, H, W)
            layer: 层索引
            
        Returns:
            Patch特征 (B, N, D) 其中N是patch数量
        """
        features = self.forward(x)
        return features[f'layer_{layer}']


@dataclass(frozen=True)
class _SAM2Config:
    config_file: str
    checkpoint_names: List[str]


_SAM2_VARIANTS: Dict[str, _SAM2Config] = {
    "tiny": _SAM2Config("sam2.1_hiera_t.yaml", ["sam2.1_hiera_tiny.pt"]),
    "small": _SAM2Config("sam2.1_hiera_s.yaml", ["sam2.1_hiera_small.pt"]),
    "base": _SAM2Config("sam2.1_hiera_b+.yaml", ["sam2.1_hiera_base_plus.pt"]),
    "large": _SAM2Config("sam2.1_hiera_l.yaml", ["sam2.1_hiera_large.pt"]),
}


class SAM2(nn.Module):
    """SAM2分割模型
    
    论文: https://arxiv.org/abs/2408.00714
    """
    
    def __init__(
        self,
        model_size: str = 'large',
        weights: Optional[str] = None,
        frozen: bool = True,
        multimask_output: bool = False,
        **kwargs
    ):
        """初始化SAM2
        
        Args:
            model_size: 模型大小 (tiny/small/base/large)
            weights: 预训练权重路径
            frozen: 是否冻结
            multimask_output: 是否输出多个掩码
        """
        super().__init__()
        
        self.model_size = model_size
        self.frozen = frozen
        self.multimask_output = multimask_output
        
        # 加载模型
        self._load_model(weights)
        
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        
        log.info(f"Initialized SAM2 ({model_size}), frozen={frozen}")
    
    def _load_model(self, weights: Optional[str]):
        """加载SAM2模型"""
        resolved_variant = _SAM2_VARIANTS.get(
            self.model_size.lower(), _SAM2_VARIANTS["large"]
        )

        checkpoint = _resolve_checkpoint(weights, resolved_variant.checkpoint_names)

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if checkpoint is None and weights:
                log.warning("SAM2 weights not found at %s", weights)

            sam2_model = build_sam2(resolved_variant.config_file, checkpoint)
            self.predictor = SAM2ImagePredictor(sam2_model)

            if checkpoint:
                log.info("Loaded SAM2 checkpoint from %s", checkpoint)

        except Exception as e:
            log.error(f"Failed to load SAM2: {e}")
            log.warning("SAM2 not available, using placeholder")
            self.predictor = None
    
    def forward(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            prompts: 提示字典，可包含:
                - boxes: (N, 4) 边界框 [x1, y1, x2, y2]
                - points: (N, 2) 点坐标
                - point_labels: (N,) 点标签 (1=前景, 0=背景)
                
        Returns:
            掩码 (B, 1, H, W)
        """
        if self.predictor is None:
            # 返回全掩码作为占位符
            return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        
        batch_size = x.shape[0]
        masks = []
        
        for i in range(batch_size):
            # 转换为numpy
            image_np = x[i].permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # 设置图像
            self.predictor.set_image(image_np)
            
            # 获取提示
            if prompts is not None:
                box = prompts.get('boxes', None)
                points = prompts.get('points', None)
                point_labels = prompts.get('point_labels', None)
            else:
                # 默认：使用图像中心点作为前景
                h, w = image_np.shape[:2]
                points = np.array([[w // 2, h // 2]])
                point_labels = np.array([1])
                box = None
            
            # 预测
            if box is not None:
                mask, _, _ = self.predictor.predict(
                    box=box[i].cpu().numpy() if isinstance(box, torch.Tensor) else box,
                    multimask_output=self.multimask_output
                )
            elif points is not None:
                mask, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=self.multimask_output
                )
            else:
                # 全图掩码
                mask = np.ones((h, w), dtype=bool)
            
            # 转换回tensor
            if self.multimask_output:
                mask = mask[0]  # 取第一个掩码
            
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            masks.append(mask_tensor)
        
        # 堆叠
        masks = torch.stack(masks, dim=0).to(x.device)
        
        return masks
    
    def segment_with_box(
        self,
        image: torch.Tensor,
        box: torch.Tensor
    ) -> torch.Tensor:
        """使用边界框进行分割
        
        Args:
            image: (3, H, W)
            box: (4,) [x1, y1, x2, y2]
            
        Returns:
            mask: (H, W)
        """
        prompts = {'boxes': box.unsqueeze(0)}
        mask = self.forward(image.unsqueeze(0), prompts)
        return mask.squeeze()


@dataclass(frozen=True)
class _DepthAnythingConfig:
    repo_id: str
    checkpoint_names: List[str]


_DEPTH_VARIANTS: Dict[str, _DepthAnythingConfig] = {
    "vits": _DepthAnythingConfig(
        repo_id="depth-anything/Depth-Anything-V2-Small",
        checkpoint_names=["depth_anything_v2_vits.pth"],
    ),
    "vitb": _DepthAnythingConfig(
        repo_id="depth-anything/Depth-Anything-V2-Base",
        checkpoint_names=["depth_anything_v2_vitb.pth"],
    ),
    "vitl": _DepthAnythingConfig(
        repo_id="depth-anything/Depth-Anything-V2-Large",
        checkpoint_names=["depth_anything_v2_vitl.pth"],
    ),
}


class DepthAnythingV2(nn.Module):
    """Depth Anything V2 深度估计模型

    论文: https://arxiv.org/abs/2401.10891
    """

    def __init__(
        self,
        encoder: str = 'vitl',
        weights: Optional[str] = None,
        frozen: bool = True,
        max_depth: float = 10.0,
        **kwargs
    ):
        super().__init__()

        self.encoder_name = encoder
        self.frozen = frozen
        self.max_depth = max_depth

        self._load_model(weights)

        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

        log.info(f"Initialized Depth Anything V2 ({encoder}), frozen={frozen}")

    def _load_model(self, weights: Optional[str]):
        resolved_variant = _DEPTH_VARIANTS.get(
            self.encoder_name.lower(), _DEPTH_VARIANTS["vitl"]
        )

        checkpoint = _resolve_checkpoint(weights, resolved_variant.checkpoint_names)

        try:
            try:
                from depth_anything_v2 import DepthAnythingV2 as DepthAnythingModel
            except ImportError:
                from depth_anything.dpt import DepthAnything as DepthAnythingModel  # type: ignore

            self.model = DepthAnythingModel.from_pretrained(resolved_variant.repo_id)

            if checkpoint:
                state_dict = torch.load(checkpoint, map_location='cpu')
                missing = self.model.load_state_dict(state_dict, strict=False)
                log.info(
                    "Loaded Depth Anything V2 weights from %s (strict=%s)",
                    checkpoint,
                    not bool(missing),
                )
            elif weights:
                log.warning("Depth Anything V2 weights not found at %s", weights)

        except Exception as e:
            log.error(f"Failed to load Depth Anything V2: {e}")
            log.warning("Using fallback depth estimator")
            self.model = self._create_fallback_model()
    
    def _create_fallback_model(self):
        """创建回退模型（简单的深度估计网络）"""
        # 使用MiDaS作为回退
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            return midas
        except:
            # 最简单的占位符
            return nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            深度图 (B, 1, H, W)，归一化到 [0, 1]
        """
        if self.frozen:
            with torch.no_grad():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """实际的前向传播"""
        # 深度预测
        depth = self.model(x)
        
        # 确保输出形状正确
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        
        # 归一化到 [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def predict_metric_depth(self, x: torch.Tensor) -> torch.Tensor:
        """预测度量深度（米）
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            深度图 (B, 1, H, W)，单位为米
        """
        normalized_depth = self.forward(x)
        metric_depth = normalized_depth * self.max_depth
        return metric_depth


class FrontendModel(nn.Module):
    """前端模型整合类
    
    将DINOv3、SAM2、Depth Anything整合到一个模块中。
    """
    
    def __init__(
        self,
        dinov3_cfg: Dict[str, Any],
        sam2_cfg: Dict[str, Any],
        depth_cfg: Dict[str, Any],
    ):
        """初始化前端模型
        
        Args:
            dinov3_cfg: DINOv3配置
            sam2_cfg: SAM2配置
            depth_cfg: Depth Anything配置
        """
        super().__init__()
        
        self.dino = DINOv3(**dinov3_cfg)
        self.sam = SAM2(**sam2_cfg)
        self.depth = DepthAnythingV2(**depth_cfg)
        
        log.info("Initialized Frontend Model")
    
    def forward(
        self,
        image: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            image: 输入图像 (B, 3, H, W)
            prompts: SAM2提示
            
        Returns:
            包含特征、掩码、深度的字典
        """
        # 提取特征
        features = self.dino(image)
        
        # 分割
        mask = self.sam(image, prompts)
        
        # 深度估计
        depth = self.depth(image)
        
        return {
            'features': features,
            'mask': mask,
            'depth': depth,
        }
    
    def extract_object_features(
        self,
        image: torch.Tensor,
        bbox: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """提取目标物体的特征
        
        Args:
            image: 输入图像 (B, 3, H, W)
            bbox: 边界框 (B, 4) [x1, y1, x2, y2]
            
        Returns:
            物体特征字典
        """
        # 分割物体
        if bbox is not None:
            mask = self.sam.segment_with_box(image[0], bbox[0])
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = self.sam(image)
        
        # 提取特征（仅物体区域）
        features = self.dino(image * mask)
        
        # 估计深度（仅物体区域）
        depth = self.depth(image)
        depth_masked = depth * mask
        
        return {
            'features': features,
            'mask': mask,
            'depth': depth_masked,
            'full_depth': depth,
        }