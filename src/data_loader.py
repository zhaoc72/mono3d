"""Utilities for loading images and videos for inference."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageSample:
    """Container representing a single RGB image sample."""

    path: str
    image: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


def load_image(
    path: str, target_size: Optional[Tuple[int, int]] = None, metadata: Optional[Dict[str, object]] = None
) -> ImageSample:
    """Load an RGB image from disk.

    Args:
        path: Path to an image readable by OpenCV.
        target_size: Optional ``(width, height)`` size to resize the image to.

    Returns:
        ``ImageSample`` containing the RGB image data.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image file: {path}")

    if target_size is not None:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageSample(path=path, image=rgb, metadata=metadata or {})


def load_video_frames(
    video_path: str,
    frame_skip: int = 1,
    target_size: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> List[ImageSample]:
    """Decode frames from a video file.

    Args:
        video_path: Path to the video file.
        frame_skip: Interval between sampled frames (``1`` means keep all frames).
        target_size: Optional ``(width, height)`` to resize decoded frames.
        max_frames: Optional cap on number of frames to load.

    Returns:
        A list of ``ImageSample`` objects representing sampled frames.
    """

    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frames: List[ImageSample] = []
    index = 0
    kept = 0

    try:
        while True:
            ret, frame_bgr = capture.read()
            if not ret:
                break

            if index % frame_skip == 0:
                if target_size is not None:
                    frame_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    ImageSample(
                        path=f"{video_path}|frame_{index}",
                        image=frame_rgb,
                        metadata={"frame_index": index, "source": video_path},
                    )
                )
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break

            index += 1
    finally:
        capture.release()

    return frames


def stream_directory_images(
    directory: str,
    valid_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg"),
    target_size: Optional[Tuple[int, int]] = None,
) -> Iterator[ImageSample]:
    """Yield ``ImageSample`` objects from a directory lazily."""

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    suffixes = tuple(s.lower() for s in valid_suffixes)
    for entry in sorted(os.listdir(directory)):
        if entry.lower().endswith(suffixes):
            yield load_image(os.path.join(directory, entry), target_size=target_size)
