"""Helpers to export segmentation outputs for downstream 3D reconstruction."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .utils import ensure_directory, save_json, save_mask


@dataclass
class ReconstructionConfig:
    output_directory: str
    save_metadata: bool = True
    metadata_keys: Optional[list[str]] = None
    mask_extension: str = ".npy"


@dataclass
class MaskArtifact:
    mask: np.ndarray
    metadata: Dict[str, object]
    stem: str


def export_mask(artifact: MaskArtifact, config: ReconstructionConfig) -> Path:
    output_dir = ensure_directory(config.output_directory)
    mask_path = output_dir / f"{artifact.stem}{config.mask_extension}"
    save_mask(mask_path, artifact.mask, format_hint=config.mask_extension.lstrip("."))
    if config.save_metadata:
        metadata_path = mask_path.with_suffix(".json")
        payload = artifact.metadata
        if config.metadata_keys:
            payload = {k: artifact.metadata.get(k) for k in config.metadata_keys}
        save_json(metadata_path, payload)
    return mask_path
