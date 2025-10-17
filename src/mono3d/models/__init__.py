"""模型模块

提供所有深度学习模型的实现，包括：
- 前端模型（DINOv3, SAM2, Depth Anything）
- 形状先验（显式和隐式）
- 3D Gaussian Splatting
- 初始化网络
"""

from .frontend import DINOv3, SAM2, DepthAnything
from .detector import GroundingDINODetector
from .shape_prior import ExplicitShapePrior, ImplicitShapePrior, ShapeVAE
from .gaussian import GaussianModel, GaussianRenderer
from .initializer import ShapeInitNet
from .losses import (
    ColorLoss,
    DepthLoss,
    MaskLoss,
    ShapePriorLoss,
    NormalSmoothLoss,
    TotalLoss,
)
from .networks import MLP, PointNetEncoder, AttentionBlock

# 自动注册所有模型
from ..registry import register

# 注册前端模型
register('model', 'dinov3')(DINOv3)
register('model', 'sam2')(SAM2)
register('model', 'depth_anything')(DepthAnything)
register('model', 'grounding_dino')(GroundingDINODetector)

# 注册形状先验
register('model', 'explicit_prior')(ExplicitShapePrior)
register('model', 'implicit_prior')(ImplicitShapePrior)
register('model', 'shape_vae')(ShapeVAE)

# 注册3DGS
register('model', 'gaussian')(GaussianModel)

# 注册初始化网络
register('model', 'shape_init_net')(ShapeInitNet)

__all__ = [
    # Frontend
    'DINOv3',
    'SAM2',
    'DepthAnything',
    'GroundingDINODetector',
    # Shape Prior
    'ExplicitShapePrior',
    'ImplicitShapePrior',
    'ShapeVAE',
    # Gaussian
    'GaussianModel',
    'GaussianRenderer',
    # Initializer
    'ShapeInitNet',
    # Losses
    'ColorLoss',
    'DepthLoss',
    'MaskLoss',
    'ShapePriorLoss',
    'NormalSmoothLoss',
    'TotalLoss',
    # Networks
    'MLP',
    'PointNetEncoder',
    'AttentionBlock',
]