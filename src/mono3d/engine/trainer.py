"""统一训练引擎"""
from typing import Dict, Any
import torch
from omegaconf import DictConfig
from pathlib import Path
import logging

from ..registry import build
from ..data.datasets import build_dataloader
from ..utils.logger import setup_logger

log = logging.getLogger(__name__)

def train(cfg: DictConfig):
    """统一训练入口"""
    
    # 设置日志
    setup_logger(cfg.paths.log_dir, cfg.logging)
    log.info(f"Starting training: {cfg.project_name}")
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # 根据任务类型选择训练器
    task = cfg.training.get("task", "shape_prior")
    
    if task == "shape_prior":
        trainer = ShapePriorTrainer(cfg, device)
    elif task == "init_net":
        trainer = InitNetTrainer(cfg, device)
    elif task == "reconstruction":
        trainer = ReconstructionTrainer(cfg, device)
    elif task == "full_pipeline":
        trainer = FullPipelineTrainer(cfg, device)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 执行训练
    trainer.train()
    
    log.info("Training completed!")


class BaseTrainer:
    """训练器基类"""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # 构建数据加载器
        self.train_loader = build_dataloader(cfg, split="train")
        self.val_loader = build_dataloader(cfg, split="val")
        
        # 创建输出目录
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建模型（子类实现）
        self.model = self.build_model()
        
        # 构建优化器
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if cfg.amp.enabled else None
    
    def build_model(self):
        """构建模型（子类实现）"""
        raise NotImplementedError
    
    def build_optimizer(self):
        """构建优化器"""
        opt_cfg = self.cfg.training.optimizer
        if opt_cfg.type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas
            )
        elif opt_cfg.type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.type}")
    
    def build_scheduler(self):
        """构建学习率调度器"""
        sch_cfg = self.cfg.training.scheduler
        if sch_cfg.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs,
                eta_min=sch_cfg.min_lr
            )
        elif sch_cfg.type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sch_cfg.step_size,
                gamma=sch_cfg.gamma
            )
        else:
            return None
    
    def train(self):
        """主训练循环"""
        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            log.info(f"Epoch {epoch}: {train_metrics}")
            
            # 验证
            if epoch % self.cfg.training.validation.interval == 0:
                val_metrics = self.validate()
                log.info(f"Validation: {val_metrics}")
                
                # 保存最佳模型
                if self.is_best(val_metrics):
                    self.save_checkpoint("best.pth")
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 定期保存
            if epoch % self.cfg.logging.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch}.pth")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch（子类实现）"""
        raise NotImplementedError
    
    def validate(self) -> Dict[str, float]:
        """验证（子类实现）"""
        raise NotImplementedError
    
    def is_best(self, metrics: Dict[str, float]) -> bool:
        """判断是否最佳"""
        metric_name = self.cfg.training.early_stopping.metric
        current = metrics.get(metric_name, float('inf'))
        
        if current < self.best_metric:
            self.best_metric = current
            return True
        return False
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.cfg,
        }, checkpoint_path)
        
        log.info(f"Checkpoint saved: {checkpoint_path}")


class ShapePriorTrainer(BaseTrainer):
    """形状先验训练器"""
    
    def build_model(self):
        """构建VAE/GAN模型"""
        return build("model", "shape_vae", **self.cfg.model.shape_prior.implicit)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            shapes = batch['shape'].to(self.device)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.cfg.amp.enabled):
                recon, mu, logvar = self.model(shapes)
                loss = self.model.loss(recon, shapes, mu, logvar)
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return {"loss": total_loss / len(self.train_loader)}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                shapes = batch['shape'].to(self.device)
                recon, mu, logvar = self.model(shapes)
                loss = self.model.loss(recon, shapes, mu, logvar)
                total_loss += loss.item()
        
        return {"loss": total_loss / len(self.val_loader)}


class InitNetTrainer(BaseTrainer):
    """初始化网络训练器"""
    
    def build_model(self):
        """构建初始化网络"""
        return build("model", "shape_init_net", device=self.device)
    
    # 实现 train_epoch 和 validate ...


class ReconstructionTrainer(BaseTrainer):
    """3DGS重建训练器"""
    
    def build_model(self):
        """构建重建模型"""
        # 组合多个模型
        models = {
            "frontend": build("model", "frontend", **self.cfg.model.frontend),
            "shape_prior": build("model", "shape_prior", **self.cfg.model.shape_prior),
            "gaussian": build("model", "gaussian", **self.cfg.model.gaussian),
        }
        return torch.nn.ModuleDict(models)
    
    # 实现 train_epoch 和 validate ...


class FullPipelineTrainer(BaseTrainer):
    """完整pipeline训练器（端到端）"""
    pass