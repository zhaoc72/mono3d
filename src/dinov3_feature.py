"""Lightweight DINOv3 feature extraction for zero-shot prompt generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
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
    output_layers: Sequence[int] = (0,)
    patch_size: int = 14
    normalize: bool = True
    torchhub_source: Optional[str] = None
    output_layer: Optional[int] = None

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

class DINOv3FeatureExtractor:
    """Wrapper that exposes patch embeddings and attention maps from DINOv3."""

    def __init__(self, config: Dinov3Config, device: torch.device | str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = self._load_model()
        self.model.eval()
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

    @torch.inference_mode()

    def extract_features(self, image: np.ndarray) -> Dict[str, object]:
        """Return patch embeddings, class token features and optional attention."""

        inputs = self._prepare(image)
        layers = self._gather_layers(inputs)
        separated = [self._split_tokens(layer) for layer in layers]
        patch_tokens = torch.stack([pair[0] for pair in separated], dim=0)  # L x B x P x D
        cls_tokens = torch.stack([pair[1] for pair in separated], dim=0)  # L x B x D
        patch_tokens = patch_tokens.mean(dim=0).squeeze(0).to("cpu")
        cls_tokens = cls_tokens.mean(dim=0).squeeze(0).to("cpu")
        num_tokens = patch_tokens.shape[0]
        grid_size = int(round(np.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"Number of patch tokens ({num_tokens}) does not form a square grid"
            )
        patch_map = patch_tokens.reshape(grid_size, grid_size, -1)

        attention = self._gather_attention(inputs)
        attention_map = None
        if attention is not None:
            # Convert attention tensor [B, heads, tokens, tokens] to a spatial heatmap.
            attn = attention.mean(dim=1)[0]  # tokens x tokens
            cls_attention = attn[0, 1:]
            attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min() + 1e-6
            )

        return {
            "patch_tokens": patch_tokens,
            "cls_token": cls_tokens,
            "grid_size": (grid_size, grid_size),
            "patch_map": patch_map,
            "attention_map": attention_map,
        }

    def to(self, device: torch.device | str) -> "DINOv3FeatureExtractor":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self


