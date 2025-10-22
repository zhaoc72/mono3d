"""DINOv3 feature extraction and saliency proposal utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from .utils import LOGGER


@dataclass
class Dinov3Config:
    repo_or_dir: str
    model_name: str
    use_torch_hub: bool = True
    torchhub_source: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_layer: int = -1


class DINOv3FeatureExtractor:
    """Wrapper around the official DINOv3 models for zero-shot proposals."""

    def __init__(self, config: Dinov3Config, device: torch.device | str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: nn.Module = self._load_model()
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.last_attention: Optional[torch.Tensor] = None
        self._supports_direct_attention = False
        self._register_attention_hook()

    def _load_model(self) -> nn.Module:
        if self.config.use_torch_hub:
            LOGGER.info(
                "Loading DINOv3 model %s from %s via torch.hub",
                self.config.model_name,
                self.config.repo_or_dir,
            )
            hub_kwargs = {
                "trust_repo": True,
                "pretrained": self.config.checkpoint_path is None,
            }
            if self.config.torchhub_source is not None:
                hub_kwargs["source"] = self.config.torchhub_source
            elif Path(self.config.repo_or_dir).exists():
                hub_kwargs["source"] = "local"
            model = torch.hub.load(
                self.config.repo_or_dir,
                self.config.model_name,
                **hub_kwargs,
            )
        else:
            import importlib

            module = importlib.import_module(self.config.repo_or_dir)
            build = getattr(module, "build_model", None)
            if build is None:
                raise AttributeError("Module must expose a build_model function when use_torch_hub=False")
            model = build(self.config.model_name)

        if self.config.checkpoint_path:
            LOGGER.info("Loading DINOv3 checkpoint from %s", self.config.checkpoint_path)
            state = torch.load(self.config.checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                LOGGER.warning("Missing keys when loading DINOv3 checkpoint: %s", missing)
            if unexpected:
                LOGGER.warning("Unexpected keys when loading DINOv3 checkpoint: %s", unexpected)

        return model.to(self.device)

    def _register_attention_hook(self) -> None:
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.last_attention = output.detach()

        self._supports_direct_attention = hasattr(self.model, "get_last_selfattention")
        if self._supports_direct_attention:
            LOGGER.debug("Using DINOv3 get_last_selfattention for attention extraction")
            return

        for name, module in self.model.named_modules():
            if "attn_drop" in name:
                module.register_forward_hook(hook_fn)
                LOGGER.debug("Registered attention hook on %s", name)
                break
        else:
            LOGGER.warning("Failed to register attention hook, attention maps may be unavailable")

    def _prepare(self, image: np.ndarray) -> torch.Tensor:
        tensor = self.preprocess(image).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        inputs = self._prepare(image)
        self.last_attention = None
        if self.device.type == "cuda":
            autocast_context = torch.cuda.amp.autocast(enabled=self.dtype == torch.float16)
        else:
            from contextlib import nullcontext

            autocast_context = nullcontext()
        with autocast_context:
            outputs = self.model(inputs)
            attention = None
            if getattr(self, "_supports_direct_attention", False):
                try:
                    attention = self.model.get_last_selfattention(inputs)
                except TypeError:
                    attention = self.model.get_last_selfattention()
            else:
                attention = self.last_attention
        if isinstance(outputs, torch.Tensor):
            feats = outputs
        elif isinstance(outputs, dict):
            key = "x_norm_clstoken" if "x_norm_clstoken" in outputs else next(iter(outputs.values()))
            feats = outputs[key]
        else:
            feats = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        return {
            "features": feats,
            "attention": attention,
        }

    @staticmethod
    def attention_to_heatmap(attention: torch.Tensor, image_size: Tuple[int, int]) -> np.ndarray:
        if attention is None:
            raise ValueError("Attention maps are not available; ensure hook registration succeeded")
        if isinstance(attention, (tuple, list)):
            attn = attention[0]
        else:
            attn = attention
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() == 3:
            attn = attn[:, 0, 1:]
        attn = attn.mean(dim=0)
        num_patches = attn.shape[0]
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise ValueError("Attention map token count is not a perfect square")
        attn_map = attn.reshape(side, side).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        resized = cv2.resize(attn_map, image_size, interpolation=cv2.INTER_CUBIC)
        return resized
