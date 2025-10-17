"""Mono3D: 单目图像与视频的目标3D重建

基于DINOv3、SAM 2和3D Gaussian Splatting的3D重建框架。
"""

__version__ = "0.1.0"
__author__ = "Mono3D Team"

# 自动注册所有模型和数据集
from .registry import (
    register,
    build,
    list_registered,
    is_registered,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY
)

# 数据模块
from .data import (
    BaseDataset,
    Pix3DDataset,
    CO3Dv2Dataset,
    ScanNetDataset,
    KITTIDataset,
    VirtualKITTIDataset,
    build_dataset,
    build_dataloader,
)

# 模型模块
from .models import (
    DINOv3,
    SAM2,
    DepthAnything,
    ExplicitShapePrior,
    ImplicitShapePrior,
    ShapeVAE,
    GaussianModel,
    ShapeInitNet,
)

# 引擎模块
from .engine import (
    train,
    infer,
    evaluate,
    BaseTrainer,
    Evaluator,
)

# 工具模块
from .utils import (
    save_pointcloud,
    load_pointcloud,
    save_mesh,
    load_mesh,
    visualize_pointcloud,
    visualize_mesh,
    compute_chamfer_distance,
    compute_iou,
    compute_fscore,
)

# 自动注册
def _auto_register():
    """自动注册所有模型和数据集"""
    from .registry import auto_register_models, auto_register_datasets
    auto_register_models()
    auto_register_datasets()

_auto_register()

__all__ = [
    # Version
    '__version__',
    # Registry
    'register',
    'build',
    'list_registered',
    'is_registered',
    'MODEL_REGISTRY',
    'DATASET_REGISTRY',
    'LOSS_REGISTRY',
    # Data
    'BaseDataset',
    'Pix3DDataset',
    'CO3Dv2Dataset',
    'ScanNetDataset',
    'KITTIDataset',
    'VirtualKITTIDataset',
    'build_dataset',
    'build_dataloader',
    # Models
    'DINOv3',
    'SAM2',
    'DepthAnything',
    'ExplicitShapePrior',
    'ImplicitShapePrior',
    'ShapeVAE',
    'GaussianModel',
    'ShapeInitNet',
    # Engine
    'train',
    'infer',
    'evaluate',
    'BaseTrainer',
    'Evaluator',
    # Utils
    'save_pointcloud',
    'load_pointcloud',
    'save_mesh',
    'load_mesh',
    'visualize_pointcloud',
    'visualize_mesh',
    'compute_chamfer_distance',
    'compute_iou',
    'compute_fscore',
]