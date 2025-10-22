"""Virtual KITTI 2 dataset helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

from ..data_loader import ImageSample, load_image


@dataclass
class VkittiFilter:
    """Filtering options when iterating over VKITTI frames."""

    scenes: Optional[Sequence[str]] = None
    clones: Optional[Sequence[str]] = None
    camera: str = "Camera_0"
    limit: Optional[int] = None


def _maybe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _intrinsics_path(clone_dir: Path, camera: str) -> Optional[str]:
    candidates = [
        clone_dir / "frames" / "intrinsic" / f"{camera}.txt",
        clone_dir / "frames" / "intrinsics" / f"{camera}.txt",
        clone_dir / f"intrinsic_{camera}.txt",
        clone_dir / f"intrinsics_{camera}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _extrinsics_path(clone_dir: Path, camera: str) -> Optional[str]:
    candidates = [
        clone_dir / "frames" / "extrinsic" / f"{camera}.txt",
        clone_dir / "frames" / "extrinsics" / f"{camera}.txt",
        clone_dir / f"extrinsic_{camera}.txt",
        clone_dir / f"extrinsics_{camera}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def iter_vkitti_frames(
    root: str | Path,
    filter: Optional[VkittiFilter] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Iterator[ImageSample]:
    """Yield frames from the Virtual KITTI 2 dataset.

    Args:
        root: Directory containing VKITTI scenes (e.g. `/media/.../vkitti2`).
        filter: Optional :class:`VkittiFilter` to restrict which scenes/clones to iterate.
        target_size: Optional ``(width, height)`` for resizing the RGB frames.

    Yields:
        ``ImageSample`` objects with metadata describing the dataset provenance.
    """

    filter = filter or VkittiFilter()
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"VKITTI root not found: {root}")

    yielded = 0
    for scene_dir in sorted(root_path.glob("Scene*")):
        scene_name = scene_dir.name
        if filter.scenes and scene_name not in filter.scenes:
            continue

        for clone_dir in sorted(scene_dir.glob("clone*")):
            clone_name = clone_dir.name
            if filter.clones and clone_name not in filter.clones:
                continue

            rgb_dir = clone_dir / "frames" / "rgb" / filter.camera
            if not rgb_dir.exists():
                continue

            depth_dir = clone_dir / "frames" / "depth" / filter.camera
            seg_dir = clone_dir / "frames" / "classSegmentation" / filter.camera

            intrinsics = _intrinsics_path(clone_dir, filter.camera)
            extrinsics = _extrinsics_path(clone_dir, filter.camera)

            for frame_path in sorted(rgb_dir.glob("*.png")):
                frame_id = frame_path.stem
                metadata: Dict[str, object] = {
                    "dataset": "vkitti2",
                    "scene": scene_name,
                    "clone": clone_name,
                    "camera": filter.camera,
                    "frame": frame_id,
                    "rgb_path": str(frame_path),
                    "depth_path": _maybe_path(depth_dir / f"{frame_id}.png") if depth_dir else None,
                    "segmentation_path": _maybe_path(seg_dir / f"{frame_id}.png") if seg_dir else None,
                    "intrinsics_path": intrinsics,
                    "extrinsics_path": extrinsics,
                }
                sample = load_image(str(frame_path), target_size=target_size, metadata=metadata)
                yield sample

                yielded += 1
                if filter.limit is not None and yielded >= filter.limit:
                    return
