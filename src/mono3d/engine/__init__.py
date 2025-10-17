"""训练和推理引擎

提供统一的训练、验证和推理接口。
"""

from .trainer import (
    BaseTrainer,
    ShapePriorTrainer,
    InitNetTrainer,
    ReconstructionTrainer,
    FullPipelineTrainer,
    train,
)

from .inferencer import (
    SingleImageInferencer,
    VideoInferencer,
    infer,
)

from .evaluator import (
    Evaluator,
    evaluate,
)

__all__ = [
    # Trainer
    'BaseTrainer',
    'ShapePriorTrainer',
    'InitNetTrainer',
    'ReconstructionTrainer',
    'FullPipelineTrainer',
    'train',
    # Inferencer
    'SingleImageInferencer',
    'VideoInferencer',
    'infer',
    # Evaluator
    'Evaluator',
    'evaluate',
]