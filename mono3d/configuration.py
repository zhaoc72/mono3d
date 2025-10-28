"""Configuration helpers for the DINOv3 pipeline launcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "加载配置需要 PyYAML，请先通过 `pip install pyyaml` 安装依赖。"
    ) from exc


@dataclass
class TaskConfig:
    """Represents a single command to execute within the pipeline."""

    name: str
    script: str
    args: List[str] = field(default_factory=list)
    use_accelerate: bool = False
    num_processes: Optional[int] = None
    mixed_precision: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    python_module: bool = False
    comment: Optional[str] = None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TaskConfig":
        required_keys = {"name", "script"}
        missing = required_keys - data.keys()
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"Task configuration is missing required keys: {missing_keys}")

        num_processes = data.get("num_processes")
        if num_processes is not None:
            num_processes = int(num_processes)

        return TaskConfig(
            name=str(data["name"]),
            script=str(data["script"]),
            args=[str(arg) for arg in data.get("args", [])],
            use_accelerate=bool(data.get("use_accelerate", False)),
            num_processes=num_processes,
            mixed_precision=data.get("mixed_precision"),
            env={str(k): str(v) for k, v in data.get("env", {}).items()},
            working_dir=str(data["working_dir"]) if data.get("working_dir") else None,
            python_module=bool(data.get("python_module", False)),
            comment=str(data["comment"]) if data.get("comment") else None,
        )


@dataclass
class Dinov3PipelineConfig:
    """Top-level configuration consumed by :class:`Dinov3PipelineRunner`."""

    repo_path: Path
    python_executable: str
    accelerate_command: str
    accelerate_config: Optional[Path]
    tasks: List[TaskConfig]
    environment: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, str] = field(default_factory=dict)
    accelerate_args: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> "Dinov3PipelineConfig":
        base_dir = base_dir or Path.cwd()
        if "repo_path" not in data:
            raise ValueError("Configuration is missing the 'repo_path' field")
        repo_path = Path(data["repo_path"])
        if not repo_path.is_absolute():
            repo_path = (base_dir / repo_path).resolve()

        accelerate_config = data.get("accelerate_config")
        if accelerate_config is not None:
            accelerate_config_path = Path(accelerate_config)
            if not accelerate_config_path.is_absolute():
                accelerate_config_path = (base_dir / accelerate_config_path).resolve()
        else:
            accelerate_config_path = None

        tasks_data = data.get("tasks", [])
        if not isinstance(tasks_data, Iterable):
            raise TypeError("tasks must be a list of task definitions")

        tasks = [TaskConfig.from_dict(task) for task in tasks_data]
        if not tasks:
            raise ValueError("At least one task must be defined in the configuration")

        environment = {str(k): str(v) for k, v in data.get("environment", {}).items()}
        context = {str(k): str(v) for k, v in data.get("context", {}).items()}
        accelerate_args = [str(arg) for arg in data.get("accelerate_args", [])]

        return Dinov3PipelineConfig(
            repo_path=repo_path,
            python_executable=str(data.get("python_executable", "python")),
            accelerate_command=str(data.get("accelerate_command", "accelerate")),
            accelerate_config=accelerate_config_path,
            tasks=tasks,
            environment=environment,
            context=context,
            accelerate_args=accelerate_args,
        )


class FormatContext(dict):
    """Dictionary with graceful fallback when formatting templates."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return ""

    @classmethod
    def build(cls, *sources: Mapping[str, Any]) -> "FormatContext":
        context: MutableMapping[str, str] = {}
        for source in sources:
            for key, value in source.items():
                context[str(key)] = str(value)
        return cls(context)


def load_pipeline_config(path: Any) -> Dinov3PipelineConfig:
    """Load a :class:`Dinov3PipelineConfig` from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Dinov3PipelineConfig.from_dict(data, base_dir=config_path.parent)
