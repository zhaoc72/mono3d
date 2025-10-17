"""统一推理引擎"""
import torch
from omegaconf import DictConfig
from pathlib import Path
import logging
from PIL import Image

from ..registry import build
from ..utils.io import save_pointcloud, save_mesh

log = logging.getLogger(__name__)

def infer(cfg: DictConfig):
    """统一推理入口"""
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # 获取输入
    input_path = cfg.get("input", cfg.get("image", None))
    if input_path is None:
        raise ValueError("Please specify input image/video path")
    
    input_path = Path(input_path)
    
    # 根据输入类型选择推理器
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        inferencer = SingleImageInferencer(cfg, device)
        result = inferencer.infer(input_path)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        inferencer = VideoInferencer(cfg, device)
        result = inferencer.infer(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    # 保存结果
    output_dir = Path(cfg.paths.output_dir) / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = input_path.stem
    if cfg.inference.export_format in ["ply", "both"]:
        save_pointcloud(result['pointcloud'], output_dir / f"{output_name}.ply")
    if cfg.inference.export_format in ["obj", "both"]:
        save_mesh(result['mesh'], output_dir / f"{output_name}.obj")
    
    log.info(f"Results saved to {output_dir}")


class SingleImageInferencer:
    """单图像推理器"""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        # 加载模型
        log.info("Loading models...")
        self.dino = build("model", "dinov3", **cfg.model.dinov3).to(device).eval()
        self.sam = build("model", "sam2", **cfg.model.sam2).to(device).eval()
        self.depth = build("model", "depth_anything", **cfg.model.depth).to(device).eval()
        self.shape_prior = build("model", "shape_prior", **cfg.model.shape_prior).to(device).eval()
        self.gaussian = build("model", "gaussian", **cfg.model.gaussian).to(device)
        
        log.info("Models loaded!")
    
    @torch.no_grad()
    def infer(self, image_path: Path):
        """执行推理"""
        log.info(f"Processing: {image_path}")
        
        # 1. 加载图像
        image = self.load_image(image_path)
        
        # 2. 提取特征
        features = self.dino(image)
        
        # 3. 分割目标
        mask = self.sam(image)
        
        # 4. 估计深度
        depth = self.depth(image)
        
        # 5. 初始化形状
        initial_shape = self.shape_prior.initialize(features, depth, mask)
        
        # 6. 优化3DGS
        gaussian_model = self.gaussian.optimize(
            image, depth, mask, initial_shape,
            iterations=self.cfg.gaussian.optimization.iterations
        )
        
        # 7. 提取结果
        result = {
            'pointcloud': gaussian_model.to_pointcloud(),
            'mesh': gaussian_model.to_mesh(),
            'gaussian': gaussian_model,
        }
        
        return result
    
    def load_image(self, path: Path):
        """加载并预处理图像"""
        image = Image.open(path).convert('RGB')
        # TODO: 添加预处理逻辑
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device)


class VideoInferencer:
    """视频推理器"""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        # TODO: 实现视频推理逻辑
    
    def infer(self, video_path: Path):
        """执行视频推理"""
        pass