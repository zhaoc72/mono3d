"""日志工具

提供日志配置和W&B集成。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logger(
    log_dir: Optional[Path] = None,
    log_config: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """配置日志系统
    
    Args:
        log_dir: 日志目录
        log_config: 日志配置
        level: 日志级别
        
    Returns:
        Logger实例
    """
    # 创建logger
    logger = logging.getLogger('mono3d')
    logger.setLevel(level)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'train_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    return logger


def get_logger(name: str = 'mono3d') -> logging.Logger:
    """获取logger实例
    
    Args:
        name: Logger名称
        
    Returns:
        Logger实例
    """
    return logging.getLogger(name)


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = ''
):
    """记录指标
    
    Args:
        metrics: 指标字典
        step: 步数
        prefix: 前缀
    """
    logger = get_logger()
    
    metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logger.info(f"{prefix}Step {step} - {metrics_str}")


class WandbLogger:
    """Weights & Biases日志记录器"""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """初始化W&B logger
        
        Args:
            project: 项目名称
            name: 运行名称
            config: 配置字典
            enabled: 是否启用
        """
        self.enabled = enabled
        
        if not enabled:
            return
        
        try:
            import wandb
            self.wandb = wandb
            
            self.run = wandb.init(
                project=project,
                name=name,
                config=config
            )
            
            log = get_logger()
            log.info(f"Initialized W&B logging: {project}/{name}")
        
        except ImportError:
            log = get_logger()
            log.warning("wandb not installed, W&B logging disabled")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标
        
        Args:
            metrics: 指标字典
            step: 可选的步数
        """
        if not self.enabled:
            return
        
        self.wandb.log(metrics, step=step)
    
    def log_image(
        self,
        key: str,
        image,
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        """记录图像
        
        Args:
            key: 键名
            image: 图像（numpy array或PIL Image）
            step: 步数
            caption: 标题
        """
        if not self.enabled:
            return
        
        self.wandb.log({
            key: self.wandb.Image(image, caption=caption)
        }, step=step)
    
    def log_pointcloud(
        self,
        key: str,
        points,
        colors: Optional = None,
        step: Optional[int] = None
    ):
        """记录点云
        
        Args:
            key: 键名
            points: 点坐标
            colors: 可选的颜色
            step: 步数
        """
        if not self.enabled:
            return
        
        self.wandb.log({
            key: self.wandb.Object3D(points)
        }, step=step)
    
    def log_model(self, filepath: Path):
        """记录模型
        
        Args:
            filepath: 模型文件路径
        """
        if not self.enabled:
            return
        
        self.wandb.save(str(filepath))
    
    def finish(self):
        """结束运行"""
        if not self.enabled:
            return
        
        self.wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(
        self,
        log_dir: Path,
        enabled: bool = True
    ):
        """初始化TensorBoard logger
        
        Args:
            log_dir: 日志目录
            enabled: 是否启用
        """
        self.enabled = enabled
        
        if not enabled:
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self.writer = SummaryWriter(log_dir=str(log_dir))
            
            log = get_logger()
            log.info(f"Initialized TensorBoard logging: {log_dir}")
        
        except ImportError:
            log = get_logger()
            log.warning("tensorboard not installed, TensorBoard logging disabled")
            self.enabled = False
    
    def log_scalar(self, key: str, value: float, step: int):
        """记录标量
        
        Args:
            key: 键名
            value: 值
            step: 步数
        """
        if not self.enabled:
            return
        
        self.writer.add_scalar(key, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """记录多个标量
        
        Args:
            metrics: 指标字典
            step: 步数
        """
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_image(self, key: str, image, step: int):
        """记录图像
        
        Args:
            key: 键名
            image: 图像tensor (C, H, W)
            step: 步数
        """
        if not self.enabled:
            return
        
        self.writer.add_image(key, image, step)
    
    def log_histogram(self, key: str, values, step: int):
        """记录直方图
        
        Args:
            key: 键名
            values: 值
            step: 步数
        """
        if not self.enabled:
            return
        
        self.writer.add_histogram(key, values, step)
    
    def close(self):
        """关闭writer"""
        if not self.enabled:
            return
        
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()