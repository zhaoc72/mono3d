"""Project-level configuration models for Mono3D orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - runtime dependency guard
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "加载配置需要 PyYAML，请先通过 `pip install pyyaml` 安装依赖。"
    ) from exc


# ---------------------------------------------------------------------------
# 基础数据模型
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """描述单个命令任务的配置。"""

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
class AccelerateOptions:
    """控制 accelerate 启动参数的配置。"""

    command: str = "accelerate"
    config_file: Optional[Path] = None
    args: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Mapping[str, Any], base_dir: Path) -> "AccelerateOptions":
        command = str(data.get("command", "accelerate"))
        config_value = data.get("config")
        if config_value:
            config_path = Path(config_value)
            if not config_path.is_absolute():
                config_path = (base_dir / config_path).resolve()
        else:
            config_path = None

        args = [str(arg) for arg in data.get("args", [])]
        return AccelerateOptions(command=command, config_file=config_path, args=args)


@dataclass
class ExecutionConfig:
    """统一定义 python 与 accelerate 的运行方式。"""

    python_executable: str = "python"
    accelerate: AccelerateOptions = field(default_factory=AccelerateOptions)

    @staticmethod
    def from_dict(data: Mapping[str, Any], base_dir: Path) -> "ExecutionConfig":
        python_exec = str(data.get("python", "python"))
        accelerate = AccelerateOptions.from_dict(data.get("accelerate", {}), base_dir)
        return ExecutionConfig(python_executable=python_exec, accelerate=accelerate)


@dataclass
class PipelineConfig:
    """描述单个推理/训练流水线。"""

    name: str
    type: str
    repo_path: Path
    enabled: bool = True
    description: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, str] = field(default_factory=dict)
    accelerate_args: List[str] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, base_dir: Path) -> "PipelineConfig":
        if "name" not in data:
            raise ValueError("Pipeline configuration is missing the 'name' field")
        if "type" not in data:
            raise ValueError("Pipeline configuration is missing the 'type' field")
        if "repo_path" not in data:
            raise ValueError("Pipeline configuration is missing the 'repo_path' field")

        repo_path = Path(data["repo_path"])
        if not repo_path.is_absolute():
            repo_path = (base_dir / repo_path).resolve()

        tasks_data = data.get("tasks", [])
        if not isinstance(tasks_data, Iterable):
            raise TypeError("Pipeline tasks must be described as a list")

        tasks = [TaskConfig.from_dict(task) for task in tasks_data]

        return PipelineConfig(
            name=str(data["name"]),
            type=str(data["type"]),
            repo_path=repo_path,
            enabled=bool(data.get("enabled", True)),
            description=str(data["description"]) if data.get("description") else None,
            environment={str(k): str(v) for k, v in data.get("environment", {}).items()},
            context={str(k): str(v) for k, v in data.get("context", {}).items()},
            accelerate_args=[str(arg) for arg in data.get("accelerate_args", [])],
            tasks=tasks,
        )


@dataclass
class ProjectConfig:
    """Mono3D 顶层配置，统一管理多条流水线。"""

    name: str
    description: Optional[str]
    environment: Dict[str, str]
    context: Dict[str, str]
    execution: ExecutionConfig
    pipelines: List[PipelineConfig]

    def get_pipeline(self, name: str) -> PipelineConfig:
        for pipeline in self.pipelines:
            if pipeline.name == name:
                return pipeline
        raise KeyError(f"Pipeline '{name}' not found in configuration")

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> "ProjectConfig":
        base_dir = base_dir or Path.cwd()
        project_meta = data.get("project", {})
        name = str(project_meta.get("name", "Mono3D Pipeline"))
        description = project_meta.get("description")
        if description is not None:
            description = str(description)

        environment = {str(k): str(v) for k, v in data.get("environment", {}).items()}
        context = {str(k): str(v) for k, v in data.get("context", {}).items()}

        execution = ExecutionConfig.from_dict(data.get("execution", {}), base_dir)

        pipelines_data = data.get("pipelines", [])
        if not pipelines_data:
            raise ValueError("At least one pipeline must be defined under 'pipelines'")
        if not isinstance(pipelines_data, Iterable):
            raise TypeError("pipelines must be described as a list")

        pipelines = [PipelineConfig.from_dict(item, base_dir=base_dir) for item in pipelines_data]

        return ProjectConfig(
            name=name,
            description=description,
            environment=environment,
            context=context,
            execution=execution,
            pipelines=pipelines,
        )


class FormatContext(dict):
    """用于字符串格式化的上下文字典。缺失键返回空字符串。"""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return ""

    @classmethod
    def build(cls, *sources: Mapping[str, Any]) -> "FormatContext":
        context: MutableMapping[str, str] = {}
        dynamic_context = cls(context)
        for source in sources:
            for key, value in source.items():
                text = str(value)
                try:
                    text = text.format_map(dynamic_context)
                except KeyError:
                    # 如果引用的键尚未定义，则保持原样，后续仍可解析。
                    pass
                context[str(key)] = text
        return dynamic_context


def load_project_config(path: Any) -> ProjectConfig:
    """从 YAML 文件加载 :class:`ProjectConfig`."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ProjectConfig.from_dict(data, base_dir=config_path.parent)
