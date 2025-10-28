"""兼容入口：转发到通用的 `run_project.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from tools.run_project import main as _main


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--config" not in argv:
        default_config = Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml"
        argv = ["--config", str(default_config)] + list(argv)
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
