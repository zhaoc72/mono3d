"""Core execution utilities for Mono3D project pipelines."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .configuration import (
    FormatContext,
    PipelineConfig,
    ProjectConfig,
    TaskConfig,
)

LOGGER = logging.getLogger(__name__)


def _default_working_directory(repo_path: Path, override: Optional[str]) -> Path:
    if override is None:
        return repo_path
    candidate = Path(override)
    if candidate.is_absolute():
        return candidate
    return (repo_path / candidate).resolve()


class ProjectPipelineRunner:
    """执行 Mono3D 配置中声明的所有流水线。"""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        self._configure_logging()

    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    # ------------------------------------------------------------------
    def run(self, selected_pipelines: Optional[Sequence[str]] = None) -> None:
        selected: Optional[set[str]] = None
        if selected_pipelines is not None:
            selected = {name for name in selected_pipelines}

        base_context = FormatContext.build(os.environ, self.config.context)

        for pipeline in self.config.pipelines:
            if not pipeline.enabled:
                LOGGER.info("[pipeline:%s] 已禁用，跳过", pipeline.name)
                continue
            if selected is not None and pipeline.name not in selected:
                LOGGER.info("[pipeline:%s] 不在 --only 指定列表内，跳过", pipeline.name)
                continue

            pipeline_context = FormatContext.build(
                base_context,
                {
                    "pipeline": pipeline.name,
                    "pipeline_type": pipeline.type,
                    "repo_path": str(pipeline.repo_path),
                },
                pipeline.context,
            )

            LOGGER.info("[pipeline:%s] 开始执行 (%s)", pipeline.name, pipeline.type)
            self._run_pipeline(pipeline, pipeline_context)

    # ------------------------------------------------------------------
    def _run_pipeline(self, pipeline: PipelineConfig, context: FormatContext) -> None:
        for task in pipeline.tasks:
            self._run_task(pipeline, task, context)

    # ------------------------------------------------------------------
    def _run_task(
        self,
        pipeline: PipelineConfig,
        task: TaskConfig,
        context: FormatContext,
    ) -> None:
        if task.comment:
            LOGGER.info("[%s:%s] %s", pipeline.name, task.name, task.comment)

        command = self._build_command(pipeline, task, context)
        env = self._build_environment(pipeline, task, context)
        working_dir = (
            task.working_dir.format_map(context) if task.working_dir is not None else None
        )
        cwd = _default_working_directory(pipeline.repo_path, working_dir)

        LOGGER.info("[%s:%s] %s", pipeline.name, task.name, shlex.join(command))
        LOGGER.debug("[%s:%s] working directory: %s", pipeline.name, task.name, cwd)
        overrides = {
            key: value
            for key, value in env.items()
            if os.environ.get(key) != value
        }
        LOGGER.debug("[%s:%s] env overrides: %s", pipeline.name, task.name, overrides)

        if self.dry_run:
            return

        subprocess.run(command, cwd=cwd, env=env, check=True)

    # ------------------------------------------------------------------
    def _build_command(
        self,
        pipeline: PipelineConfig,
        task: TaskConfig,
        context: FormatContext,
    ) -> List[str]:
        command: List[str]
        if task.use_accelerate:
            accelerate = self.config.execution.accelerate
            command = [accelerate.command, "launch"]
            if accelerate.config_file is not None:
                command += ["--config_file", str(accelerate.config_file)]
            if task.num_processes is not None:
                command += ["--num_processes", str(task.num_processes)]
            if task.mixed_precision:
                command += ["--mixed_precision", task.mixed_precision]
            if accelerate.args:
                command += self._format_items(accelerate.args, context)
            if pipeline.accelerate_args:
                command += self._format_items(pipeline.accelerate_args, context)
            command += self._script_invocation(pipeline, task, context)
        else:
            command = [self.config.execution.python_executable]
            command += self._script_invocation(pipeline, task, context)

        command += self._format_items(task.args, context)
        return command

    # ------------------------------------------------------------------
    def _script_invocation(
        self,
        pipeline: PipelineConfig,
        task: TaskConfig,
        context: FormatContext,
    ) -> List[str]:
        if task.python_module:
            return ["-m", task.script.format_map(context)]

        script_value = task.script.format_map(context)
        script_path = Path(script_value)
        if not script_path.is_absolute():
            script_path = (pipeline.repo_path / script_path).resolve()
        return [str(script_path)]

    # ------------------------------------------------------------------
    def _build_environment(
        self,
        pipeline: PipelineConfig,
        task: TaskConfig,
        context: FormatContext,
    ) -> dict:
        env = dict(os.environ)
        env.update(self._format_mapping(self.config.environment.items(), context))
        env.update(self._format_mapping(pipeline.environment.items(), context))
        env.update(self._format_mapping(task.env.items(), context))
        return env

    # ------------------------------------------------------------------
    @staticmethod
    def _format_items(items: Iterable[str], context: FormatContext) -> List[str]:
        return [str(item).format_map(context) for item in items]

    # ------------------------------------------------------------------
    @staticmethod
    def _format_mapping(
        items: Iterable[Tuple[str, str]],
        context: FormatContext,
    ) -> dict:
        formatted = {}
        for key, value in items:
            formatted[str(key)] = str(value).format_map(context)
        return formatted
