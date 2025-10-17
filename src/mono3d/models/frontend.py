"""前端模型：DINOv3, SAM2, Depth Anything

封装预训练的视觉基础模型用于特征提取、分割和深度估计。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)


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
        try:
            # 尝试使用torch.hub
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                self.backbone_name,
                pretrained=True if weights is None else False
            )
            
            # 如果提供了自定义权重
            if weights is not None:
                state_dict = torch.load(weights, map_location='cpu')
                self.model.load_state_dict(state_dict)
                log.info(f"Loaded custom weights from {weights}")
        
        except Exception as e:
            log.error(f"Failed to load DINOv3: {e}")
            # 回退方案：创建一个简单的特征提取器
            log.warning("Using fallback feature extractor")
            self.model = self._create_fallback_model()
    
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
        features = {}
        
        # 如果是DINOv2模型
        if hasattr(self.model, 'get_intermediate_layers'):
            # 获取多层特征
            intermediate = self.model.get_intermediate_layers(
                x,
                n=self.output_layers,
                return_class_token=True
            )
            
            for i, (layer_idx, feat) in enumerate(zip(self.output_layers, intermediate)):
                features[f'layer_{layer_idx}'] = feat
            
            # 最后一层作为全局特征
            features['global'] = intermediate[-1]
        
        else:
            # 回退模型
            feat = self.model(x)
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
        try:
            # 尝试导入SAM2库
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # 模型配置文件
            config_map = {
                'tiny': 'sam2_hiera_t.yaml',
                'small': 'sam2_hiera_s.yaml',
                'base': 'sam2_hiera_b+.yaml',
                'large': 'sam2_hiera_l.yaml',
            }
            
            config_file = config_map.get(self.model_size, 'sam2_hiera_l.yaml')
            
            # 构建模型
            if weights is None:
                # 使用默认权重
                checkpoint = f"checkpoints/{config_file.replace('.yaml', '.pt')}"
            else:
                checkpoint = weights
            
            sam2_model = build_sam2(config_file, checkpoint)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            log.info(f"Loaded SAM2 from {checkpoint}")
        
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


class DepthAnything(nn.Module):
    """Depth Anything深度估计模型
    
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
        """初始化Depth Anything
        
        Args:
            encoder: 编码器大小 (vits/vitb/vitl)
            weights: 预训练权重路径
            frozen: 是否冻结
            max_depth: 最大深度值（米）
        """
        super().__init__()
        
        self.encoder_name = encoder
        self.frozen = frozen
        self.max_depth = max_depth
        
        # 加载模型
        self._load_model(weights)
        
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        
        log.info(f"Initialized Depth Anything ({encoder}), frozen={frozen}")
    
    def _load_model(self, weights: Optional[str]):
        """加载Depth Anything模型"""
        try:
            # 尝试导入Depth Anything
            from depth_anything.dpt import DepthAnything as DepthAnythingModel
            
            encoder_map = {
                'vits': 'vits',
                'vitb': 'vitb',
                'vitl': 'vitl',
            }
            
            encoder = encoder_map.get(self.encoder_name, 'vitl')
            
            self.model = DepthAnythingModel.from_pretrained(
                f'LiheYoung/depth_anything_{encoder}14'
            )
            
            if weights is not None:
                state_dict = torch.load(weights, map_location='cpu')
                self.model.load_state_dict(state_dict)
                log.info(f"Loaded custom weights from {weights}")
        
        except Exception as e:
            log.error(f"Failed to load Depth Anything: {e}")
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
        self.depth = DepthAnything(**depth_cfg)
        
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