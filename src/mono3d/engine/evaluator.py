"""评估引擎

提供模型评估和指标计算功能。
"""

from typing import Dict, Any, Optional, List
import torch
from omegaconf import DictConfig
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

from ..registry import build
from ..data.datasets import build_dataloader
from ..utils.metrics import (
    compute_chamfer_distance,
    compute_iou,
    compute_fscore,
    compute_psnr,
    compute_ssim,
    evaluate_reconstruction
)
from ..utils.io import save_pointcloud, save_mesh
from ..utils.visualization import (
    visualize_pointcloud,
    save_comparison_image
)

log = logging.getLogger(__name__)


def evaluate(cfg: DictConfig):
    """评估入口
    
    Args:
        cfg: 配置对象
    """
    log.info("Starting evaluation")
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # 创建评估器
    evaluator = Evaluator(cfg, device)
    
    # 执行评估
    results = evaluator.evaluate()
    
    # 保存结果
    evaluator.save_results(results)
    
    log.info("Evaluation completed!")


class Evaluator:
    """评估器"""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        """初始化评估器
        
        Args:
            cfg: 配置对象
            device: 设备
        """
        self.cfg = cfg
        self.device = device
        
        # 创建输出目录
        self.output_dir = Path(cfg.paths.output_dir) / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建数据加载器
        log.info("Building dataloader...")
        self.test_loader = build_dataloader(cfg, split="test", shuffle=False)
        log.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        # 加载模型
        log.info("Loading models...")
        self.models = self.load_models()
        
        # 设置为评估模式
        for model in self.models.values():
            if hasattr(model, 'eval'):
                model.eval()
    
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """加载所有需要的模型"""
        models = {}
        
        # 前端模型
        if hasattr(self.cfg.model, 'frontend'):
            from ..models.frontend import FrontendModel
            models['frontend'] = FrontendModel(
                self.cfg.model.dinov3,
                self.cfg.model.sam2,
                self.cfg.model.depth
            ).to(self.device)
        
        # 形状先验
        if hasattr(self.cfg.model, 'shape_prior'):
            models['shape_prior'] = build(
                'model',
                'shape_vae',
                **self.cfg.model.shape_prior.implicit
            ).to(self.device)
            
            # 加载权重
            if self.cfg.model.shape_prior.get('weights'):
                checkpoint = torch.load(
                    self.cfg.model.shape_prior.weights,
                    map_location=self.device
                )
                models['shape_prior'].load_state_dict(checkpoint['model_state_dict'])
        
        # Gaussian模型
        if hasattr(self.cfg.model, 'gaussian'):
            from ..models.gaussian import GaussianModel
            models['gaussian'] = GaussianModel(**self.cfg.model.gaussian).to(self.device)
            
            # 加载权重
            if self.cfg.model.gaussian.get('weights'):
                from ..utils.io import load_gaussian_model
                models['gaussian'] = load_gaussian_model(
                    self.cfg.model.gaussian.weights,
                    self.device
                )
        
        return models
    
    def evaluate(self) -> Dict[str, Any]:
        """执行评估
        
        Returns:
            评估结果字典
        """
        log.info("Running evaluation...")
        
        all_metrics = {
            'chamfer': [],
            'precision': [],
            'recall': [],
            'fscore': [],
            'psnr': [],
            'ssim': [],
        }
        
        sample_results = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # 执行重建
                result = self.reconstruct_sample(batch)
                
                # 计算指标
                metrics = self.compute_metrics(result, batch)
                
                # 累积指标
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
                
                # 保存样本结果
                sample_results.append({
                    'sample_id': idx,
                    'metrics': metrics,
                    'result': result
                })
                
                # 可视化部分样本
                if idx < 10:  # 只保存前10个样本的可视化
                    self.visualize_sample(result, batch, idx)
        
        # 计算平均指标
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        # 汇总结果
        results = {
            'average_metrics': avg_metrics,
            'per_sample_metrics': all_metrics,
            'sample_results': sample_results,
            'config': dict(self.cfg)
        }
        
        # 打印结果
        log.info("\n" + "=" * 50)
        log.info("Evaluation Results:")
        log.info("=" * 50)
        for key, value in avg_metrics.items():
            log.info(f"{key}: {value:.4f}")
        log.info("=" * 50)
        
        return results
    
    def reconstruct_sample(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """重建单个样本
        
        Args:
            batch: 批次数据
            
        Returns:
            重建结果
        """
        # 获取输入
        image = batch['image'].to(self.device)
        
        # 1. 前端处理
        if 'frontend' in self.models:
            frontend_output = self.models['frontend'](image)
            features = frontend_output['features']
            mask = frontend_output['mask']
            depth = frontend_output['depth']
        else:
            # 使用预计算的特征
            features = batch.get('features', None)
            mask = batch.get('mask', None)
            depth = batch.get('depth', None)
            
            if features is not None:
                features = features.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            if depth is not None:
                depth = depth.to(self.device)
        
        # 2. 初始化形状（使用先验）
        if 'shape_prior' in self.models:
            # 这里简化处理：直接从先验采样
            with torch.no_grad():
                z = torch.randn(1, self.models['shape_prior'].latent_dim).to(self.device)
                initial_shape = self.models['shape_prior'].decode(z)
        else:
            # 使用深度生成初始点云
            if depth is not None:
                from ..data.utils import depth_to_pointcloud, CameraParams
                
                # 创建默认相机参数
                h, w = depth.shape[2:]
                camera = CameraParams(
                    fx=525.0, fy=525.0,
                    cx=w/2, cy=h/2,
                    R=np.eye(3),
                    t=np.zeros(3),
                    width=w, height=h
                )
                
                depth_np = depth[0, 0].cpu().numpy()
                mask_np = mask[0, 0].cpu().numpy() if mask is not None else None
                
                pc = depth_to_pointcloud(depth_np, camera, mask_np)
                initial_shape = torch.from_numpy(pc['points']).float().unsqueeze(0).to(self.device)
            else:
                # 随机初始化
                initial_shape = torch.randn(1, 2048, 3).to(self.device)
        
        # 3. 3DGS重建（简化版本）
        if 'gaussian' in self.models:
            # 初始化高斯
            self.models['gaussian'].initialize_from_pointcloud(
                initial_shape[0],
                colors=None,
                normals=None
            )
            
            # 优化（这里跳过，使用初始化结果）
            # 实际应用中需要运行优化循环
            
            # 转换为点云
            reconstructed_pc = self.models['gaussian'].to_pointcloud()
            
            result = {
                'points': reconstructed_pc['points'],
                'colors': reconstructed_pc['colors'],
                'gaussian_model': self.models['gaussian'],
            }
        else:
            # 直接使用初始形状
            result = {
                'points': initial_shape[0].cpu().numpy(),
                'colors': None,
            }
        
        return result
    
    def compute_metrics(
        self,
        result: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算指标
        
        Args:
            result: 重建结果
            batch: 批次数据（包含真值）
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 获取预测和真值
        pred_points = torch.from_numpy(result['points']).float()
        
        if 'gt_points' in batch:
            gt_points = batch['gt_points'][0]  # (N, 3)
        elif 'shape' in batch:
            gt_points = batch['shape'][0]
        else:
            # 没有真值，跳过
            log.warning("No ground truth available for metric computation")
            return metrics
        
        # Chamfer距离
        metrics['chamfer'] = compute_chamfer_distance(
            pred_points.unsqueeze(0),
            gt_points.unsqueeze(0)
        )
        
        # F-score
        fscore_results = compute_fscore(
            pred_points.unsqueeze(0),
            gt_points.unsqueeze(0),
            threshold=0.01
        )
        metrics.update(fscore_results)
        
        # 如果有渲染图像，计算图像指标
        if 'rendered_image' in result and 'image' in batch:
            pred_image = result['rendered_image']
            gt_image = batch['image'][0].cpu().permute(1, 2, 0).numpy()
            
            # 确保范围正确
            if pred_image.max() <= 1.0:
                pred_image = (pred_image * 255).astype(np.uint8)
            if gt_image.max() <= 1.0:
                gt_image = (gt_image * 255).astype(np.uint8)
            
            metrics['psnr'] = compute_psnr(pred_image, gt_image)
            metrics['ssim'] = compute_ssim(pred_image, gt_image)
        
        return metrics
    
    def visualize_sample(
        self,
        result: Dict[str, Any],
        batch: Dict[str, Any],
        sample_id: int
    ):
        """可视化样本
        
        Args:
            result: 重建结果
            batch: 批次数据
            sample_id: 样本ID
        """
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 可视化点云
        points = result['points']
        colors = result.get('colors', None)
        
        visualize_pointcloud(
            points,
            colors=colors,
            save_path=vis_dir / f"sample_{sample_id:04d}_pointcloud.png",
            show=False,
            title=f"Sample {sample_id}"
        )
        
        # 如果有真值，也可视化
        if 'gt_points' in batch or 'shape' in batch:
            gt_points = batch.get('gt_points', batch.get('shape'))[0].cpu().numpy()
            
            visualize_pointcloud(
                gt_points,
                save_path=vis_dir / f"sample_{sample_id:04d}_ground_truth.png",
                show=False,
                title=f"Sample {sample_id} - Ground Truth"
            )
        
        # 保存对比图
        if 'image' in batch:
            images = [batch['image'][0].cpu().permute(1, 2, 0).numpy()]
            labels = ['Input']
            
            if 'rendered_image' in result:
                images.append(result['rendered_image'])
                labels.append('Reconstruction')
            
            if 'gt_image' in batch:
                images.append(batch['gt_image'][0].cpu().permute(1, 2, 0).numpy())
                labels.append('Ground Truth')
            
            save_comparison_image(
                images,
                labels,
                vis_dir / f"sample_{sample_id:04d}_comparison.png"
            )
    
    def save_results(self, results: Dict[str, Any]):
        """保存评估结果
        
        Args:
            results: 评估结果
        """
        # 保存为JSON
        import json
        
        # 准备可序列化的结果
        serializable_results = {
            'average_metrics': results['average_metrics'],
            'config': results['config']
        }
        
        json_path = self.output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        log.info(f"Results saved to {json_path}")
        
        # 保存为CSV（每个样本的指标）
        per_sample_metrics = results['per_sample_metrics']
        
        df = pd.DataFrame(per_sample_metrics)
        csv_path = self.output_dir / "per_sample_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        log.info(f"Per-sample metrics saved to {csv_path}")
        
        # 保存汇总报告
        self.save_report(results)
    
    def save_report(self, results: Dict[str, Any]):
        """生成并保存评估报告
        
        Args:
            results: 评估结果
        """
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("3D Reconstruction Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 配置信息
            f.write("Configuration:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Dataset: {self.cfg.data.name}\n")
            f.write(f"Test samples: {len(self.test_loader.dataset)}\n")
            f.write("\n")
            
            # 平均指标
            f.write("Average Metrics:\n")
            f.write("-" * 60 + "\n")
            avg_metrics = results['average_metrics']
            for key, value in sorted(avg_metrics.items()):
                f.write(f"{key:20s}: {value:.6f}\n")
            f.write("\n")
            
            # 统计信息
            f.write("Statistics:\n")
            f.write("-" * 60 + "\n")
            for key in ['chamfer', 'fscore', 'psnr', 'ssim']:
                if key in results['per_sample_metrics']:
                    values = results['per_sample_metrics'][key]
                    if values:
                        f.write(f"{key}:\n")
                        f.write(f"  Mean: {np.mean(values):.6f}\n")
                        f.write(f"  Std:  {np.std(values):.6f}\n")
                        f.write(f"  Min:  {np.min(values):.6f}\n")
                        f.write(f"  Max:  {np.max(values):.6f}\n")
                        f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        log.info(f"Evaluation report saved to {report_path}")


class BenchmarkEvaluator(Evaluator):
    """基准测试评估器
    
    用于在多个数据集上进行基准测试。
    """
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        
        # 基准测试数据集列表
        self.benchmark_datasets = cfg.get('benchmark_datasets', [cfg.data.name])
    
    def evaluate(self) -> Dict[str, Any]:
        """在所有基准数据集上评估"""
        all_results = {}
        
        for dataset_name in self.benchmark_datasets:
            log.info(f"\n{'='*60}")
            log.info(f"Evaluating on {dataset_name}")
            log.info('='*60)
            
            # 临时修改配置
            original_dataset = self.cfg.data.name
            self.cfg.data.name = dataset_name
            
            # 重新构建数据加载器
            self.test_loader = build_dataloader(self.cfg, split="test", shuffle=False)
            
            # 运行评估
            results = super().evaluate()
            
            # 保存结果
            dataset_output_dir = self.output_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)
            
            # 恢复配置
            self.cfg.data.name = original_dataset
            
            all_results[dataset_name] = results
        
        # 生成对比报告
        self.save_benchmark_report(all_results)
        
        return all_results
    
    def save_benchmark_report(self, all_results: Dict[str, Dict[str, Any]]):
        """保存基准测试对比报告
        
        Args:
            all_results: 所有数据集的结果
        """
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Benchmark Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # 创建对比表格
            datasets = list(all_results.keys())
            metrics = ['chamfer', 'fscore', 'psnr', 'ssim']
            
            f.write(f"{'Dataset':<20}")
            for metric in metrics:
                f.write(f"{metric:<15}")
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            for dataset in datasets:
                avg_metrics = all_results[dataset]['average_metrics']
                f.write(f"{dataset:<20}")
                
                for metric in metrics:
                    value = avg_metrics.get(metric, 0.0)
                    f.write(f"{value:<15.6f}")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        log.info(f"Benchmark report saved to {report_path}")
        
        # 同时保存为CSV
        rows = []
        for dataset in datasets:
            row = {'dataset': dataset}
            row.update(all_results[dataset]['average_metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Benchmark results saved to {csv_path}")


class IncrementalEvaluator(Evaluator):
    """增量评估器
    
    在训练过程中定期评估模型。
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        checkpoint_dir: Path
    ):
        super().__init__(cfg, device)
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def evaluate_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """评估所有检查点
        
        Returns:
            每个检查点的评估结果
        """
        # 找到所有检查点
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pth"))
        
        if not checkpoints:
            log.warning(f"No checkpoints found in {self.checkpoint_dir}")
            return {}
        
        all_results = {}
        
        for checkpoint_path in checkpoints:
            log.info(f"\nEvaluating checkpoint: {checkpoint_path.name}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 更新模型权重
            if 'gaussian' in self.models:
                self.models['gaussian'].load_state_dict(
                    checkpoint['model_state_dict']
                )
            
            # 评估
            results = super().evaluate()
            
            # 保存结果
            epoch = checkpoint['epoch']
            all_results[f"epoch_{epoch}"] = results
        
        # 生成趋势报告
        self.save_trend_report(all_results)
        
        return all_results
    
    def save_trend_report(self, all_results: Dict[str, Dict[str, Any]]):
        """保存训练趋势报告
        
        Args:
            all_results: 所有检查点的结果
        """
        import matplotlib.pyplot as plt
        
        # 提取指标趋势
        epochs = []
        metrics_over_time = {
            'chamfer': [],
            'fscore': [],
            'psnr': [],
            'ssim': []
        }
        
        for checkpoint_name, results in sorted(all_results.items()):
            epoch = int(checkpoint_name.split('_')[1])
            epochs.append(epoch)
            
            avg_metrics = results['average_metrics']
            for metric in metrics_over_time:
                value = avg_metrics.get(metric, 0.0)
                metrics_over_time[metric].append(value)
        
        # 绘制趋势图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, values) in enumerate(metrics_over_time.items()):
            axes[idx].plot(epochs, values, marker='o')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} over Training')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        trend_path = self.output_dir / "training_trends.png"
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info(f"Training trends saved to {trend_path}")