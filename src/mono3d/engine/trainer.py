"""统一训练引擎"""

from typing import Dict, Any, Optional
import torch
from omegaconf import DictConfig
from pathlib import Path
import logging
from tqdm import tqdm
import time

from omegaconf import OmegaConf

from ..registry import build
from ..data.datasets import build_dataloader
from ..utils.logger import setup_logger, WandbLogger, TensorBoardLogger
from ..utils.metrics import evaluate_reconstruction
from ..models.losses import TotalLoss

log = logging.getLogger(__name__)


def train(cfg: DictConfig):
    """统一训练入口
    
    Args:
        cfg: 配置对象
    """
    # 设置日志
    setup_logger(cfg.paths.log_dir, cfg.logging)
    log.info(f"Starting training: {cfg.project_name}")
    log.info(f"Config: {cfg}")
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
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
    try:
        trainer.train()
        log.info("Training completed successfully!")
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted.pth")
    except Exception as e:
        log.error(f"Training failed with error: {e}", exc_info=True)
        raise


class BaseTrainer:
    """训练器基类"""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        """初始化训练器
        
        Args:
            cfg: 配置对象
            device: 设备
        """
        self.cfg = cfg
        self.device = device
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

        # ------------------------------------------------------------------
        # Resolve optional configuration sections with sensible defaults so
        # the trainer can be instantiated in lightweight unit tests that only
        # provide a subset of the full Hydra configuration.
        # ------------------------------------------------------------------
        default_paths = OmegaConf.create({
            "output_dir": "./outputs",
            "checkpoint_dir": "./outputs/checkpoints",
            "log_dir": "./outputs/logs",
        })
        self.paths = OmegaConf.merge(
            default_paths,
            cfg.get("paths") or OmegaConf.create({})
        )

        default_logging = OmegaConf.create({
            "use_wandb": False,
            "wandb_project": "mono3d",
        })
        self.logging_cfg = OmegaConf.merge(
            default_logging,
            cfg.get("logging") or OmegaConf.create({})
        )

        default_amp = OmegaConf.create({"enabled": False})
        self.amp_cfg = OmegaConf.merge(
            default_amp,
            cfg.get("amp") or OmegaConf.create({})
        )

        default_training = OmegaConf.create({
            "epochs": 1,
            "optimizer": {
                "type": "adam",
                "lr": 1e-4,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
            },
            "scheduler": {"type": "none"},
            "validation": {"interval": 1},
            "early_stopping": {
                "enabled": False,
                "metric": "loss",
                "mode": "min",
                "patience": 5,
            },
            "gradient_clip": 0.0,
        })
        self.training_cfg = OmegaConf.merge(
            default_training,
            cfg.get("training") or OmegaConf.create({})
        )

        # 创建输出目录
        self.output_dir = Path(self.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(self.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建数据加载器
        log.info("Building dataloaders...")
        self.train_loader = build_dataloader(cfg, split="train")
        self.val_loader = build_dataloader(cfg, split="val")
        log.info(f"Train samples: {len(self.train_loader.dataset)}")
        log.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        # 构建模型（子类实现）
        log.info("Building model...")
        self.model = self.build_model()
        self.model.to(device)
        
        # 构建优化器和调度器
        log.info("Building optimizer...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # AMP
        self.use_amp = bool(self.amp_cfg.enabled)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 日志记录器
        self.setup_loggers()

        # 早停
        self.early_stopping_patience = self.training_cfg.early_stopping.patience
        self.early_stopping_counter = 0
        
        log.info("Trainer initialized")
    
    def setup_loggers(self):
        """设置日志记录器"""
        self.loggers = []
        
        # W&B
        if bool(self.logging_cfg.use_wandb):
            wandb_logger = WandbLogger(
                project=self.logging_cfg.wandb_project,
                name=f"{self.cfg.project_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                config=dict(self.cfg),
                enabled=True
            )
            self.loggers.append(wandb_logger)

        # TensorBoard
        tb_logger = TensorBoardLogger(
            log_dir=self.paths.log_dir,
            enabled=True
        )
        self.loggers.append(tb_logger)
    
    def build_model(self):
        """构建模型（子类实现）"""
        raise NotImplementedError
    
    def build_optimizer(self):
        """构建优化器"""
        opt_cfg = self.training_cfg.optimizer
        
        if opt_cfg.type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.get("betas", (0.9, 0.999))
            )
        elif opt_cfg.type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.get("betas", (0.9, 0.999))
            )
        elif opt_cfg.type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.type}")
    
    def build_scheduler(self):
        """构建学习率调度器"""
        sch_cfg = self.training_cfg.scheduler
        
        if sch_cfg.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_cfg.epochs,
                eta_min=sch_cfg.min_lr
            )
        elif sch_cfg.type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sch_cfg.step_size,
                gamma=sch_cfg.gamma
            )
        elif sch_cfg.type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sch_cfg.factor,
                patience=sch_cfg.patience
            )
        elif sch_cfg.type == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sch_cfg.type}")
    
    def train(self):
        """主训练循环"""
        log.info("=" * 50)
        log.info("Starting training")
        log.info("=" * 50)
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.training_cfg.epochs):
            self.current_epoch = epoch
            
            log.info(f"\nEpoch {epoch + 1}/{self.training_cfg.epochs}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 记录
            self.log_metrics(train_metrics, prefix="train/")
            log.info(f"Train - {self._format_metrics(train_metrics)}")
            
            # 验证
            if (epoch + 1) % self.training_cfg.validation.interval == 0:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, prefix="val/")
                log.info(f"Val - {self._format_metrics(val_metrics)}")
                
                # 保存最佳模型
                if self.is_best(val_metrics):
                    log.info("New best model!")
                    self.save_checkpoint("best.pth")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # 早停检查
                if self.training_cfg.early_stopping.enabled:
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        log.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # 定期保存
            if (epoch + 1) % self.cfg.logging.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")
        
        # 保存最终模型
        self.save_checkpoint("final.pth")
        
        # 关闭日志记录器
        for logger in self.loggers:
            if hasattr(logger, 'close'):
                logger.close()
            elif hasattr(logger, 'finish'):
                logger.finish()
        
        log.info("Training finished!")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch（子类实现）"""
        raise NotImplementedError
    
    def validate(self) -> Dict[str, float]:
        """验证（子类实现）"""
        raise NotImplementedError
    
    def is_best(self, metrics: Dict[str, float]) -> bool:
        """判断是否最佳"""
        metric_name = self.training_cfg.early_stopping.metric
        mode = self.training_cfg.early_stopping.mode
        
        current = metrics.get(metric_name, float('inf'))
        
        if mode == 'min':
            is_better = current < self.best_metric
        else:  # max
            is_better = current > self.best_metric
        
        if is_better:
            self.best_metric = current
            return True
        
        return False
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': dict(self.cfg),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        log.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """记录指标到所有日志记录器"""
        # 添加前缀
        metrics_with_prefix = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # 添加学习率
        if self.optimizer is not None:
            metrics_with_prefix[f"{prefix}lr"] = self.optimizer.param_groups[0]['lr']
        
        # 记录到所有logger
        for logger in self.loggers:
            if hasattr(logger, 'log_scalars'):
                logger.log_scalars(metrics_with_prefix, self.global_step)
            elif hasattr(logger, 'log'):
                logger.log(metrics_with_prefix, self.global_step)
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化指标为字符串"""
        return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])


class ShapePriorTrainer(BaseTrainer):
    """形状先验训练器（VAE/GAN）"""
    
    def build_model(self):
        """构建VAE模型"""
        from ..models.shape_prior import ShapeVAE
        return ShapeVAE(**self.cfg.model.shape_prior.implicit)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 获取形状数据
            if 'shape' in batch:
                shapes = batch['shape'].to(self.device)
            elif 'points' in batch:
                shapes = batch['points'].to(self.device)
            else:
                raise ValueError("Batch must contain 'shape' or 'points'")
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(shapes)
                
                # 计算损失
                loss = self.model.loss(
                    output['reconstructed'],
                    shapes,
                    output['mu'],
                    output['logvar'],
                    kld_weight=self.cfg.model.shape_prior.implicit.loss.kld_weight
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.training_cfg.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_cfg.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.training_cfg.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_cfg.gradient_clip
                    )
                
                self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if 'shape' in batch:
                    shapes = batch['shape'].to(self.device)
                else:
                    shapes = batch['points'].to(self.device)
                
                output = self.model(shapes)
                
                loss = self.model.loss(
                    output['reconstructed'],
                    shapes,
                    output['mu'],
                    output['logvar'],
                    kld_weight=self.cfg.model.shape_prior.implicit.loss.kld_weight
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {'loss': avg_loss}


class InitNetTrainer(BaseTrainer):
    """初始化网络训练器"""
    
    def build_model(self):
        """构建初始化网络"""
        from ..models.initializer import ShapeInitNet
        return ShapeInitNet(**self.cfg.model.initializer)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # 获取特征和目标
            features = batch['features'].to(self.device)
            target_shapes = batch['shape'].to(self.device)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(features)
                
                # 计算损失（Chamfer距离）
                if 'points' in pred:
                    pred_shapes = pred['points']
                else:
                    # 需要解码latent
                    raise NotImplementedError("Latent decoding not implemented")
                
                from ..models.losses import chamfer_distance
                loss = chamfer_distance(pred_shapes, target_shapes)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        return {'loss': total_loss / len(self.train_loader)}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                target_shapes = batch['shape'].to(self.device)
                
                pred = self.model(features)
                pred_shapes = pred['points']
                
                from ..models.losses import chamfer_distance
                loss = chamfer_distance(pred_shapes, target_shapes)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / len(self.val_loader)}


class ReconstructionTrainer(BaseTrainer):
    """3DGS重建训练器"""
    
    def build_model(self):
        """构建重建模型"""
        # 组合多个模型
        from ..models.frontend import FrontendModel
        from ..models.shape_prior import ImplicitShapePrior
        from ..models.gaussian import GaussianModel
        
        models = torch.nn.ModuleDict({
            "frontend": FrontendModel(
                self.cfg.model.dinov3,
                self.cfg.model.sam2,
                self.cfg.model.depth
            ),
            "shape_prior": ImplicitShapePrior(**self.cfg.model.shape_prior.implicit),
            "gaussian": GaussianModel(**self.cfg.model.gaussian),
        })
        
        return models
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        # 3DGS训练逻辑较复杂，这里提供简化版本
        self.model['gaussian'].train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            depths = batch.get('depth', None)
            masks = batch.get('mask', None)
            
            if depths is not None:
                depths = depths.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            
            # 渲染
            rendered = self.model['gaussian'].render()
            
            # 计算损失
            from ..models.losses import compute_loss
            losses = compute_loss(
                rendered,
                {'image': images, 'depth': depths, 'mask': masks},
                self.cfg.loss_weights
            )
            
            loss = losses['total']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        return {'loss': total_loss / len(self.train_loader)}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model['gaussian'].eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                depths = batch.get('depth', None)
                masks = batch.get('mask', None)
                
                if depths is not None:
                    depths = depths.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                
                rendered = self.model['gaussian'].render()
                
                from ..models.losses import compute_loss
                losses = compute_loss(
                    rendered,
                    {'image': images, 'depth': depths, 'mask': masks},
                    self.cfg.loss_weights
                )
                
                total_loss += losses['total'].item()
        
        return {'loss': total_loss / len(self.val_loader)}


class FullPipelineTrainer(BaseTrainer):
    """完整pipeline训练器（端到端）"""
    
    def build_model(self):
        """构建完整pipeline"""
        # 实现端到端训练逻辑
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        # 实现完整训练流程
        pass
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        pass