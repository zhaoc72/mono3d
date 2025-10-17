#!/usr/bin/env python3
"""
Mono3D: 单目3D重建统一CLI
用法:
    python scripts/mono3d.py train +experiment=pix3d_reconstruction
    python scripts/mono3d.py infer image=path/to/image.jpg
    python scripts/mono3d.py cache +data=pix3d
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from mono3d.registry import list_registered

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """主入口，根据命令分发到不同模块"""
    
    # 打印配置（调试用）
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))
        return
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    
    # 根据任务类型分发
    command = cfg.get("command", "train")
    
    if command == "train":
        from mono3d.engine.trainer import train
        train(cfg)
    
    elif command == "infer":
        from mono3d.engine.inferencer import infer
        infer(cfg)
    
    elif command == "eval":
        from mono3d.engine.evaluator import evaluate
        evaluate(cfg)
    
    elif command == "cache":
        from mono3d.data.cache import build_cache
        build_cache(cfg)
    
    elif command == "export":
        from tools.export_models import export
        export(cfg)
    
    elif command == "list":
        # 列出已注册的模型
        print("Registered models:")
        for name in list_registered("model"):
            print(f"  - {name}")
    
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()