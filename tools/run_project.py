"""命令行入口：按照 YAML 配置执行 Mono3D 项目流水线。"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

from mono3d.core import ProjectPipelineRunner, load_project_config


def parse_overrides(values: Sequence[str]) -> list[tuple[str, str]]:
    overrides: list[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"无效的 --set 参数：{item!r}，正确格式为 key=value")
        key, value = item.split("=", 1)
        overrides.append((key.strip(), value))
    return overrides


def apply_overrides(config, overrides: Iterable[tuple[str, str]]) -> None:
    for key, value in overrides:
        if key.startswith("pipeline:"):
            remainder = key[len("pipeline:") :]
            pipeline_name, _, context_key = remainder.partition(".")
            if not pipeline_name or not context_key:
                raise ValueError(
                    f"无效的流水线覆盖项：{key!r}，示例：pipeline:dinov3.backbone_checkpoint=/path/to.pth"
                )
            pipeline = config.get_pipeline(pipeline_name)
            pipeline.context[context_key] = value
        else:
            config.context[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Mono3D 项目流水线")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划执行的命令，不真正运行",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细的调试日志",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="仅运行指定名称的流水线 (可多个)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置中的 context 值，支持 pipeline:名称.key=value",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用流水线并退出",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_project_config(args.config)

    if args.list:
        print("当前配置中的流水线：")
        for pipeline in config.pipelines:
            status = "启用" if pipeline.enabled else "禁用"
            desc = f" ({pipeline.description})" if pipeline.description else ""
            print(f"- {pipeline.name} [{pipeline.type}] - {status}{desc}")
        return 0

    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    try:
        apply_overrides(config, overrides)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
        return 2

    runner = ProjectPipelineRunner(config, dry_run=args.dry_run, verbose=args.verbose)
    runner.run(args.only)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
