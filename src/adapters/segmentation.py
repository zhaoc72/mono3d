"""Segmentation adapter operating on DINOv3 patch maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pipeline_types import SegmentationOutput
from ..utils import LOGGER
from .loading import load_adapter_checkpoint


@dataclass
class SegmentationAdapterConfig:
    """Runtime configuration for the simplified segmentation head."""

    checkpoint_path: str = ""
    feature_dim: int = 1024
    num_classes: int = 150
    class_names: Optional[Sequence[str]] = None


def build_segmentation_adapter(
    config: SegmentationAdapterConfig,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
) -> "SegmentationAdapter":
    """Instantiate the lightweight segmentation adapter."""

    checkpoint_path = config.checkpoint_path
    LOGGER.info(
        "Building segmentation adapter (checkpoint: %s)", checkpoint_path or "<random>"
    )

    adapter = SegmentationAdapter(
        feature_dim=config.feature_dim,
        num_classes=config.num_classes,
        class_names=config.class_names,
        device=device,
        dtype=torch_dtype,
    )

    if checkpoint_path:
        loaded = load_adapter_checkpoint(adapter.model, checkpoint_path, "Segmentation")
        if loaded == 0:
            LOGGER.info("Segmentation adapter will operate with randomly initialized weights")
    else:
        LOGGER.info("Segmentation adapter: no checkpoint provided, using randomly initialized weights")

    adapter.model = adapter.model.to(device=device, dtype=torch_dtype)
    adapter.model.eval()

    return adapter


def build_ade20k_adapter(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> "SegmentationAdapter":
    """Backward-compatible helper mirroring the old API."""

    config = SegmentationAdapterConfig(
        checkpoint_path=checkpoint_path,
        feature_dim=kwargs.get("feature_dim", 1024),
        num_classes=kwargs.get("num_classes", 150),
    )
    return build_segmentation_adapter(config, device=device, torch_dtype=torch_dtype)


class SimpleSegmentationHead(nn.Module):
    """Simplified linear segmentation head."""
    
    def __init__(self, feature_dim: int = 1024, num_classes: int = 150):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Simple linear classifier
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        
    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [B, H, W, D] patch features
        
        Returns:
            logits: [B, C, H, W]
        """
        # Permute to [B, D, H, W]
        x = features.permute(0, 3, 1, 2)
        
        # Apply classifier
        logits = self.classifier(x)  # [B, C, H, W]
        
        return logits


class SegmentationAdapter:
    """Wrapper for segmentation model."""
    
    # ADE20K class names (150 classes)
    ADE20K_CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
        'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
        'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
        'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
        'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
        'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
        'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
        'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
        'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
        'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
        'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
        'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
        'television receiver', 'airplane', 'dirt track', 'apparel', 'pole',
        'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
        'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
        'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
        'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step',
        'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
        'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
        'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
    ]
    
    def __init__(
        self,
        feature_dim: int = 1024,
        num_classes: int = 150,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.model = SimpleSegmentationHead(feature_dim, num_classes)
        self.device = torch.device(device)
        self.dtype = dtype
        if class_names is None:
            class_names = self.ADE20K_CLASSES[:num_classes]
        self.class_names = list(class_names)
        self.num_classes = num_classes

    def _ensure_input_dim(self, feature_dim: int) -> None:
        """Ensure the classifier expects the provided feature dimension."""

        current_dim = getattr(self.model, "feature_dim", feature_dim)
        if feature_dim == current_dim:
            return

        LOGGER.info(
            "Adapting segmentation adapter input dimension from %d to %d",
            current_dim,
            feature_dim,
        )

        new_classifier = nn.Conv2d(feature_dim, self.model.num_classes, kernel_size=1)
        nn.init.xavier_uniform_(new_classifier.weight)
        if new_classifier.bias is not None:
            nn.init.zeros_(new_classifier.bias)
        self.model.classifier = new_classifier.to(device=self.device, dtype=self.dtype)
        self.model.feature_dim = feature_dim

    @torch.inference_mode()
    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> SegmentationOutput:
        """
        Run semantic segmentation on patch tokens.
        
        Args:
            patch_tokens: Patch tokens from DINOv3 backbone [N, D]
            image_size: Original image size (width, height)
            grid_size: Grid size (height, width)
            
        Returns:
            SegmentationOutput with logits [C, H, W], class_names, and activation
        """
        # Convert to tensor
        tokens_tensor = torch.from_numpy(patch_tokens).to(
            device=self.device, dtype=self.dtype
        )

        # Reshape to [1, H, W, D] for segmentation head
        grid_h, grid_w = grid_size
        feature_dim = tokens_tensor.shape[-1]
        self._ensure_input_dim(feature_dim)
        tokens_reshaped = tokens_tensor.reshape(grid_h, grid_w, feature_dim)
        tokens_batch = tokens_reshaped.unsqueeze(0)  # [1, H, W, D]
        
        # Run segmentation
        try:
            logits = self.model(tokens_batch)  # [1, C, H, W]
        except Exception as e:
            LOGGER.warning(
                f"Segmentation failed: {e}, returning empty predictions"
            )
            return SegmentationOutput(
                logits=np.zeros((self.num_classes, grid_h, grid_w), dtype=np.float32),
                class_names=self.class_names,
                activation="sigmoid",
            )
        
        # Remove batch dimension and convert to numpy
        logits_np = logits.squeeze(0).cpu().numpy()  # [C, H, W]
        
        LOGGER.debug(
            f"Segmentation: logits shape {logits_np.shape}, "
            f"min={logits_np.min():.3f}, max={logits_np.max():.3f}"
        )
        
        return SegmentationOutput(
            logits=logits_np,
            class_names=self.class_names,
            activation="sigmoid",  # ADE20K uses sigmoid activation
        )