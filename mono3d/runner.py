"""Pipeline runner that orchestrates subprocess execution."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .configuration import Dinov3PipelineConfig, FormatContext, TaskConfig

LOGGER = logging.getLogger(__name__)


def _default_working_directory(repo_path: Path, override: Optional[str]) -> Path:
    if override is None:
        return repo_path
    candidate = Path(override)
    if candidate.is_absolute():
        return candidate
    return (repo_path / candidate).resolve()


class Dinov3PipelineRunner:
    """Execute DINOv3 tasks based on a :class:`Dinov3PipelineConfig`."""

    def __init__(
        self,
        config: Dinov3PipelineConfig,
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        self._configure_logging()

    def _configure_logging(self) -> None:
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    def run(self) -> None:
        base_context = FormatContext.build(
            os.environ,
            {
                "repo_path": str(self.config.repo_path),
                "python_executable": self.config.python_executable,
                "accelerate_command": self.config.accelerate_command,
            },
            self.config.context,
        )

        for task in self.config.tasks:
            self._run_task(task, base_context)

    # ------------------------------------------------------------------
    def _run_task(self, task: TaskConfig, context: FormatContext) -> None:
        if task.comment:
            LOGGER.info("[%s] %s", task.name, task.comment)

        command = self._build_command(task, context)
        env = self._build_environment(task, context)
        working_dir = (
            task.working_dir.format_map(context)
            if task.working_dir is not None
            else None
        )
        cwd = _default_working_directory(self.config.repo_path, working_dir)

        LOGGER.info("[%s] %s", task.name, shlex.join(command))
        LOGGER.debug("[%s] working directory: %s", task.name, cwd)
        overrides = {
            key: value
            for key, value in env.items()
            if os.environ.get(key) != value
        }
        LOGGER.debug("[%s] env overrides: %s", task.name, overrides)

        if self.dry_run:
            return

        subprocess.run(command, cwd=cwd, env=env, check=True)

    # ------------------------------------------------------------------
    def _build_command(self, task: TaskConfig, context: FormatContext) -> List[str]:
        if task.use_accelerate:
            command: List[str] = [self.config.accelerate_command, "launch"]
            if self.config.accelerate_config is not None:
                command += ["--config_file", str(self.config.accelerate_config)]
            if task.num_processes is not None:
                command += ["--num_processes", str(task.num_processes)]
            if task.mixed_precision:
                command += ["--mixed_precision", task.mixed_precision]
            if self.config.accelerate_args:
                command += self._format_items(self.config.accelerate_args, context)
            command += self._script_invocation(task, context)
        else:
            command = [self.config.python_executable]
            command += self._script_invocation(task, context)

        command += self._format_items(task.args, context)
        return command

    def _script_invocation(self, task: TaskConfig, context: FormatContext) -> List[str]:
        if task.python_module:
            return ["-m", task.script.format_map(context)]

        script_value = task.script.format_map(context)
        script_path = Path(script_value)
        if not script_path.is_absolute():
            script_path = (self.config.repo_path / script_path).resolve()
        return [str(script_path)]

    def _build_environment(self, task: TaskConfig, context: FormatContext) -> dict:
        env = dict(os.environ)
        env.update(self._format_mapping(self.config.environment.items(), context))
        env.update(self._format_mapping(task.env.items(), context))
        return env

    @staticmethod
    def _format_items(items: Iterable[str], context: FormatContext) -> List[str]:
        return [str(item).format_map(context) for item in items]

    @staticmethod
    def _format_mapping(
        items: Iterable[Tuple[str, str]],
        context: FormatContext,
    ) -> dict:
        formatted = {}
        for key, value in items:
            formatted[str(key)] = str(value).format_map(context)
        return formatted
