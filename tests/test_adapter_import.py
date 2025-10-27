"""Tests for adapter import path discovery helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_original_cv2 = sys.modules.get("cv2")
if _original_cv2 is None:
    sys.modules["cv2"] = types.ModuleType("cv2")

from src.__main__ import _instantiate_adapter


def test_instantiate_adapter_discovers_projects_namespace(tmp_path: Path) -> None:
    """Adapters located under a projects/ directory should be discoverable."""

    repo_root = tmp_path / "dinov3"
    adapters_dir = repo_root / "projects" / "dinov3" / "adapters"
    adapters_dir.mkdir(parents=True)

    (repo_root / "projects" / "dinov3" / "__init__.py").write_text("")
    (adapters_dir / "__init__.py").write_text("")
    (adapters_dir / "detection.py").write_text(
        "def build_coco_adapter(**kwargs):\n    return kwargs\n"
    )

    original_sys_path = list(sys.path)
    try:
        adapter_cfg = {
            "target": "dinov3.adapters.detection:build_coco_adapter",
            "python_paths": [str(repo_root)],
            "kwargs": {"sentinel": True},
        }
        adapter = _instantiate_adapter(adapter_cfg, device="cpu", dtype="float32")
    finally:
        sys.path[:] = original_sys_path
        for module_name in [
            "dinov3",
            "dinov3.adapters",
            "dinov3.adapters.detection",
        ]:
            sys.modules.pop(module_name, None)
        if _original_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = _original_cv2

    assert adapter["sentinel"] is True
