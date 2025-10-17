"""数据集实现

包含所有支持的数据集：Pix3D, CO3Dv2, ScanNet, KITTI, Virtual KITTI等。
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import json
import pickle
import logging

from omegaconf import OmegaConf

from .base import BaseDataset, MultiViewDataset, VideoDataset
from .utils import CameraParams, read_camera_params, depth_to_pointcloud
from .transforms import get_default_transforms

log = logging.getLogger(__name__)


# ============================================================================
# Pix3D Dataset
# ============================================================================

class Pix3DDataset(BaseDataset):
    """Pix3D数据集
    
    单张图像 + 3D模型对齐
    论文: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Pix3D_Dataset_and_CVPR_2018_paper.pdf
    """
    
    CATEGORIES = [
        'bed', 'bookcase', 'chair', 'desk', 'misc',
        'sofa', 'table', 'tool', 'wardrobe'
    ]
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.categories_filter = categories
        super().__init__(root, split, transform, cache_dir, use_cache)
    
    def _load_samples(self) -> list:
        """加载Pix3D样本"""
        # 读取标注文件
        anno_file = self.root / "pix3d.json"
        
        if not anno_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {anno_file}\n"
                f"Please download Pix3D dataset from: "
                f"http://pix3d.csail.mit.edu/"
            )
        
        with open(anno_file) as f:
            annotations = json.load(f)
        
        # 过滤样本
        samples = []
        
        for anno in annotations:
            # 类别过滤
            category = anno['category']
            if self.categories_filter and category not in self.categories_filter:
                continue
            
            # 分割过滤（基于truncated字段）
            # train: truncated=False, test: truncated=True
            # 简化处理：随机划分
            img_path = self.root / anno['img']
            img_id = img_path.stem
            hash_val = hash(img_id) % 100
            
            if self.split == 'train' and hash_val >= 80:
                continue
            elif self.split == 'val' and (hash_val < 70 or hash_val >= 80):
                continue
            elif self.split == 'test' and hash_val < 70:
                continue
            
            samples.append({
                'image_id': img_id,
                'image_path': img_path,
                'mask_path': self.root / anno['mask'],
                'model_path': self.root / anno['model'],
                'category': category,
                'truncated': anno.get('truncated', False),
                'occluded': anno.get('occluded', False),
                'bbox': anno.get('bbox'),  # [x, y, w, h]
                'pose': anno.get('rot_mat'),  # 旋转矩阵
                'trans': anno.get('trans_mat'),  # 平移向量
                'focal_length': anno.get('focal_length', 1500.0),
                'inplane_rotation': anno.get('inplane_rotation', 0.0),
            })
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载单个样本"""
        meta = self.samples[idx]
        
        # 加载图像
        image = Image.open(meta['image_path']).convert('RGB')
        
        # 加载掩码
        mask = Image.open(meta['mask_path']).convert('L')
        
        # 加载3D模型（可选，用于训练形状先验）
        # 这里只返回路径，实际加载在需要时进行
        
        sample = {
            'image': image,
            'mask': mask,
            'image_id': meta['image_id'],
            'category': meta['category'],
            'model_path': str(meta['model_path']),
            'bbox': meta['bbox'],
            'focal_length': meta['focal_length'],
        }
        
        # 构造相机参数
        h, w = image.size[1], image.size[0]
        fx = fy = meta['focal_length']
        cx, cy = w / 2, h / 2
        
        R = np.array(meta['pose']) if meta['pose'] else np.eye(3)
        t = np.array(meta['trans']) if meta['trans'] else np.zeros(3)
        
        sample['camera'] = CameraParams(fx, fy, cx, cy, R, t, w, h)
        
        return sample
    
    def load_3d_model(self, idx: int) -> Dict[str, np.ndarray]:
        """加载3D模型（用于训练）
        
        Returns:
            字典包含:
                - vertices: (V, 3)
                - faces: (F, 3)
        """
        meta = self.samples[idx]
        model_path = meta['model_path']
        
        # 支持OBJ格式
        if model_path.suffix == '.obj':
            return self._load_obj(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def _load_obj(self, filepath: Path) -> Dict[str, np.ndarray]:
        """加载OBJ文件"""
        vertices = []
        faces = []
        
        with open(filepath) as f:
            for line in f:
                if line.startswith('v '):
                    # 顶点
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    # 面（注意OBJ索引从1开始）
                    parts = line.strip().split()
                    face = []
                    for part in parts[1:]:
                        # 处理 "v/vt/vn" 格式
                        v_idx = int(part.split('/')[0]) - 1
                        face.append(v_idx)
                    faces.append(face)
        
        return {
            'vertices': np.array(vertices, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
        }


# ============================================================================
# CO3Dv2 Dataset
# ============================================================================

class CO3Dv2Dataset(MultiViewDataset):
    """CO3Dv2多视角数据集
    
    50个类别，19,000个物体，150万帧
    论文: https://openaccess.thecvf.com/content/ICCV2021/papers/Reizenstein_Common_Objects_in_3D_Large-Scale_Learning_and_Evaluation_of_Real-Life_ICCV_2021_paper.pdf
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        category: Optional[str] = None,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
        num_views: int = 10,  # 每个物体采样的视角数
    ):
        self.category = category
        self.num_views = num_views
        super().__init__(root, split, transform, cache_dir, use_cache)
    
    def _load_samples(self) -> list:
        """加载CO3Dv2样本"""
        samples = []
        
        # CO3D按类别组织
        if self.category:
            categories = [self.category]
        else:
            # 列出所有类别目录
            categories = [d.name for d in self.root.iterdir() if d.is_dir()]
        
        for cat in categories:
            cat_dir = self.root / cat
            
            # 读取序列列表
            split_file = cat_dir / f"set_lists/set_lists_{self.split}.json"
            
            if not split_file.exists():
                log.warning(f"Split file not found: {split_file}")
                continue
            
            with open(split_file) as f:
                sequence_names = json.load(f)
            
            # 遍历序列
            for seq_name in sequence_names:
                seq_dir = cat_dir / seq_name
                
                if not seq_dir.exists():
                    continue
                
                # 读取帧标注
                frame_anno_file = seq_dir / "frame_annotations.jgz"
                if not frame_anno_file.exists():
                    continue
                
                # 简化：假设每个序列是一个样本
                samples.append({
                    'sequence_name': seq_name,
                    'category': cat,
                    'sequence_dir': seq_dir,
                    'frame_anno_file': frame_anno_file,
                })
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载单个序列的参考视图"""
        meta = self.samples[idx]
        # 读取帧标注
        import gzip
        with gzip.open(meta['frame_anno_file'], 'rt') as f:
            frame_annotations = json.load(f)
        
        # 选择参考帧（第一帧）
        ref_frame = frame_annotations[0]
        
        # 加载图像
        image_path = meta['sequence_dir'] / 'images' / ref_frame['image']['path']
        image = Image.open(image_path).convert('RGB')
        
        # 加载掩码
        mask_path = meta['sequence_dir'] / 'masks' / ref_frame['mask']['path']
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
        else:
            # 创建全掩码
            mask = Image.new('L', image.size, 255)
        
        # 构造相机参数
        camera_data = ref_frame['viewpoint']
        focal_length = camera_data['focal_length']
        principal_point = camera_data['principal_point']
        
        w, h = image.size
        fx = fy = focal_length[0] * w  # 归一化焦距转像素焦距
        cx = principal_point[0] * w
        cy = principal_point[1] * h
        
        # 外参
        R = np.array(camera_data['R']).reshape(3, 3)
        T = np.array(camera_data['T'])
        
        sample = {
            'image': image,
            'mask': mask,
            'image_id': ref_frame['frame_number'],
            'sequence_name': meta['sequence_name'],
            'category': meta['category'],
            'camera': CameraParams(fx, fy, cx, cy, R, T, w, h),
        }
        
        return sample
    
    def get_views(self, idx: int) -> Tuple[List[Image.Image], List[CameraParams]]:
        """获取某个序列的多个视角"""
        meta = self.samples[idx]
        
        # 读取所有帧
        import gzip
        with gzip.open(meta['frame_anno_file'], 'rt') as f:
            frame_annotations = json.load(f)
        
        # 采样视角
        num_frames = len(frame_annotations)
        if num_frames <= self.num_views:
            indices = list(range(num_frames))
        else:
            # 均匀采样
            indices = np.linspace(0, num_frames - 1, self.num_views, dtype=int)
        
        images = []
        cameras = []
        
        for i in indices:
            frame = frame_annotations[i]
            
            # 加载图像
            image_path = meta['sequence_dir'] / 'images' / frame['image']['path']
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            
            # 构造相机
            camera_data = frame['viewpoint']
            focal_length = camera_data['focal_length']
            principal_point = camera_data['principal_point']
            
            w, h = image.size
            fx = fy = focal_length[0] * w
            cx = principal_point[0] * w
            cy = principal_point[1] * h
            
            R = np.array(camera_data['R']).reshape(3, 3)
            T = np.array(camera_data['T'])
            
            cameras.append(CameraParams(fx, fy, cx, cy, R, T, w, h))
        
        return images, cameras
    
    def get_camera_params(self, idx: int, view_idx: int) -> Dict[str, Any]:
        """获取指定视角的相机参数"""
        _, cameras = self.get_views(idx)
        return cameras[view_idx]


# ============================================================================
# ScanNet Dataset
# ============================================================================

class ScanNetDataset(VideoDataset):
    """ScanNet室内场景数据集
    
    1500+个RGB-D扫描序列
    网站: http://www.scan-net.org/
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        scene_types: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
        load_depth: bool = True,
    ):
        self.scene_types = scene_types
        self.load_depth = load_depth
        super().__init__(root, split, transform, cache_dir, use_cache)
    
    def _load_samples(self) -> list:
        """加载ScanNet场景"""
        # 读取分割文件
        split_file = self.root / f"scannetv2_{self.split}.txt"
        
        if not split_file.exists():
            # 如果没有官方分割，使用所有场景
            scenes = [d.name for d in self.root.iterdir() 
                     if d.is_dir() and d.name.startswith('scene')]
        else:
            with open(split_file) as f:
                scenes = [line.strip() for line in f]
        
        samples = []
        
        for scene_name in scenes:
            scene_dir = self.root / scene_name
            
            if not scene_dir.exists():
                continue
            
            # 检查必要文件
            color_dir = scene_dir / 'color'
            depth_dir = scene_dir / 'depth'
            pose_dir = scene_dir / 'pose'
            
            if not color_dir.exists():
                continue
            
            # 获取帧列表
            frame_files = sorted(color_dir.glob('*.jpg'))
            
            if len(frame_files) == 0:
                continue
            
            samples.append({
                'scene_name': scene_name,
                'scene_dir': scene_dir,
                'num_frames': len(frame_files),
                'color_dir': color_dir,
                'depth_dir': depth_dir,
                'pose_dir': pose_dir,
            })
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载场景的第一帧作为参考"""
        meta = self.samples[idx]
        
        # 加载第一帧
        frame_files = sorted(meta['color_dir'].glob('*.jpg'))
        image = Image.open(frame_files[0]).convert('RGB')
        
        # 加载深度
        depth = None
        if self.load_depth and meta['depth_dir'].exists():
            depth_file = meta['depth_dir'] / (frame_files[0].stem + '.png')
            if depth_file.exists():
                depth = np.array(Image.open(depth_file), dtype=np.float32)
                depth = depth / 1000.0  # mm -> m
        
        # 加载位姿
        pose_file = meta['pose_dir'] / (frame_files[0].stem + '.txt')
        if pose_file.exists():
            pose = np.loadtxt(pose_file)
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = np.eye(3)
            t = np.zeros(3)
        
        # ScanNet相机内参（固定）
        w, h = image.size
        fx = fy = 577.870605  # 默认内参
        cx = w / 2
        cy = h / 2
        
        sample = {
            'image': image,
            'depth': depth,
            'scene_name': meta['scene_name'],
            'frame_idx': 0,
            'camera': CameraParams(fx, fy, cx, cy, R, t, w, h),
        }
        
        return sample
    
    def get_frames(self, idx: int, num_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取场景的多帧"""
        meta = self.samples[idx]
        
        frame_files = sorted(meta['color_dir'].glob('*.jpg'))
        
        if num_frames is not None:
            # 采样帧
            total = len(frame_files)
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        
        for frame_file in frame_files:
            # 加载图像
            image = Image.open(frame_file).convert('RGB')
            
            # 加载深度
            depth = None
            if self.load_depth and meta['depth_dir'].exists():
                depth_file = meta['depth_dir'] / (frame_file.stem + '.png')
                if depth_file.exists():
                    depth = np.array(Image.open(depth_file), dtype=np.float32)
                    depth = depth / 1000.0
            
            # 加载位姿
            pose_file = meta['pose_dir'] / (frame_file.stem + '.txt')
            if pose_file.exists():
                pose = np.loadtxt(pose_file)
                R = pose[:3, :3]
                t = pose[:3, 3]
            else:
                R = np.eye(3)
                t = np.zeros(3)
            
            w, h = image.size
            fx = fy = 577.870605
            cx = w / 2
            cy = h / 2
            
            frames.append({
                'image': image,
                'depth': depth,
                'frame_idx': int(frame_file.stem),
                'camera': CameraParams(fx, fy, cx, cy, R, t, w, h),
            })
        
        return frames
    
    def get_frame_timestamps(self, idx: int) -> List[float]:
        """获取帧时间戳（ScanNet没有时间戳，返回帧索引）"""
        meta = self.samples[idx]
        frame_files = sorted(meta['color_dir'].glob('*.jpg'))
        return [float(f.stem) for f in frame_files]


# ============================================================================
# KITTI Dataset
# ============================================================================

class KITTIDataset(VideoDataset):
    """KITTI自动驾驶数据集
    
    主要用于户外车辆重建
    网站: http://www.cvlibs.net/datasets/kitti/
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        sequence: Optional[str] = None,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.sequence = sequence
        super().__init__(root, split, transform, cache_dir, use_cache)
    
    def _load_samples(self) -> list:
        """加载KITTI序列"""
        # KITTI Odometry数据集结构
        sequences_dir = self.root / 'sequences'
        
        if not sequences_dir.exists():
            raise FileNotFoundError(f"Sequences directory not found: {sequences_dir}")
        
        # 定义训练/测试分割（Odometry benchmark）
        if self.split == 'train':
            seq_ids = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
        elif self.split == 'val':
            seq_ids = ['09']
        elif self.split == 'test':
            seq_ids = ['10']
        else:
            # 使用所有序列
            seq_ids = [d.name for d in sequences_dir.iterdir() if d.is_dir()]
        
        # 如果指定了序列，只使用该序列
        if self.sequence:
            seq_ids = [self.sequence]
        
        samples = []
        
        for seq_id in seq_ids:
            seq_dir = sequences_dir / seq_id
            
            if not seq_dir.exists():
                continue
            
            # 图像目录
            image_dir = seq_dir / 'image_2'  # 左彩色相机
            
            if not image_dir.exists():
                continue
            
            # 获取帧数
            frame_files = sorted(image_dir.glob('*.png'))
            
            samples.append({
                'sequence_id': seq_id,
                'sequence_dir': seq_dir,
                'image_dir': image_dir,
                'num_frames': len(frame_files),
            })
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载序列的第一帧"""
        meta = self.samples[idx]
        
        # 加载第一帧
        frame_files = sorted(meta['image_dir'].glob('*.png'))
        image = Image.open(frame_files[0]).convert('RGB')
        
        # 加载相机标定
        calib_file = meta['sequence_dir'] / 'calib.txt'
        calib = self._load_calibration(calib_file)
        
        # 加载位姿
        poses = self._load_poses(meta['sequence_dir'] / 'poses.txt')
        pose = poses[0] if len(poses) > 0 else np.eye(4)
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        w, h = image.size
        
        sample = {
            'image': image,
            'sequence_id': meta['sequence_id'],
            'frame_idx': 0,
            'camera': CameraParams(
                calib['fx'], calib['fy'],
                calib['cx'], calib['cy'],
                R, t, w, h
            ),
        }
        
        return sample
    
    def _load_calibration(self, calib_file: Path) -> Dict[str, float]:
        """加载KITTI标定文件"""
        calib = {}
        
        with open(calib_file) as f:
            for line in f:
                if line.startswith('P2:'):
                    # 左彩色相机投影矩阵
                    values = [float(x) for x in line.strip().split()[1:]]
                    P2 = np.array(values).reshape(3, 4)
                    
                    calib['fx'] = P2[0, 0]
                    calib['fy'] = P2[1, 1]
                    calib['cx'] = P2[0, 2]
                    calib['cy'] = P2[1, 2]
                    break
        
        return calib
    
    def _load_poses(self, poses_file: Path) -> List[np.ndarray]:
        """加载位姿文件"""
        if not poses_file.exists():
            return []
        
        poses = []
        
        with open(poses_file) as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                pose = np.array(values).reshape(3, 4)
                # 转换为4x4齐次矩阵
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                poses.append(pose_4x4)
        
        return poses
    
    def get_frames(self, idx: int, num_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取序列的多帧"""
        meta = self.samples[idx]
        
        frame_files = sorted(meta['image_dir'].glob('*.png'))
        
        if num_frames is not None:
            total = len(frame_files)
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # 加载标定和位姿
        calib_file = meta['sequence_dir'] / 'calib.txt'
        calib = self._load_calibration(calib_file)
        
        poses = self._load_poses(meta['sequence_dir'] / 'poses.txt')
        
        frames = []
        
        for i, frame_file in enumerate(frame_files):
            image = Image.open(frame_file).convert('RGB')
            
            frame_idx = int(frame_file.stem)
            pose = poses[frame_idx] if frame_idx < len(poses) else np.eye(4)
            
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            w, h = image.size
            
            frames.append({
                'image': image,
                'frame_idx': frame_idx,
                'camera': CameraParams(
                    calib['fx'], calib['fy'],
                    calib['cx'], calib['cy'],
                    R, t, w, h
                ),
            })
        
        return frames
    
    def get_frame_timestamps(self, idx: int) -> List[float]:
        """获取帧时间戳"""
        meta = self.samples[idx]
        
        times_file = meta['sequence_dir'] / 'times.txt'
        
        if not times_file.exists():
            # 返回帧索引
            return list(range(meta['num_frames']))
        
        with open(times_file) as f:
            return [float(line.strip()) for line in f]


# ============================================================================
# Virtual KITTI Dataset
# ============================================================================

class VirtualKITTIDataset(VideoDataset):
    """Virtual KITTI 2合成数据集
    
    5个场景，完整标注（深度、分割、3D边界框等）
    网站: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        scene: Optional[str] = None,
        variant: str = "clone",  # clone / fog / rain / sunset
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.scene = scene
        self.variant = variant
        super().__init__(root, split, transform, cache_dir, use_cache)
    
    def _load_samples(self) -> list:
        """加载Virtual KITTI样本"""
        # Virtual KITTI 2结构: vkitti_2.0.3/{Scene}/{variant}/
        
        # 场景列表
        if self.scene:
            scenes = [self.scene]
        else:
            scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        
        samples = []
        
        for scene in scenes:
            scene_dir = self.root / scene / self.variant
            
            if not scene_dir.exists():
                continue
            
            # 图像目录
            image_dir = scene_dir / 'frames' / 'rgb' / 'Camera_0'
            
            if not image_dir.exists():
                continue
            
            frame_files = sorted(image_dir.glob('*.jpg'))
            
            # 简单分割：前80%训练，后20%测试
            total = len(frame_files)
            if self.split == 'train':
                frame_files = frame_files[:int(0.8 * total)]
            elif self.split == 'test':
                frame_files = frame_files[int(0.8 * total):]
            
            samples.append({
                'scene': scene,
                'variant': self.variant,
                'scene_dir': scene_dir,
                'image_dir': image_dir,
                'num_frames': len(frame_files),
            })
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载第一帧"""
        meta = self.samples[idx]
        
        frame_files = sorted(meta['image_dir'].glob('*.jpg'))
        image = Image.open(frame_files[0]).convert('RGB')
        
        # 加载深度
        depth_dir = meta['scene_dir'] / 'frames' / 'depth' / 'Camera_0'
        depth_file = depth_dir / (frame_files[0].stem + '.png')
        
        if depth_file.exists():
            depth = np.array(Image.open(depth_file), dtype=np.float32)
            depth = depth / 100.0  # cm -> m
        else:
            depth = None
        
        # 加载相机参数（从intrinsic.txt和extrinsic.txt）
        intrinsic_file = meta['scene_dir'] / 'intrinsic.txt'
        extrinsic_file = meta['scene_dir'] / 'extrinsic.txt'
        
        # 简化：使用默认参数
        w, h = image.size
        fx = fy = 725.0
        cx = w / 2
        cy = h / 2
        R = np.eye(3)
        t = np.zeros(3)
        
        sample = {
            'image': image,
            'depth': depth,
            'scene': meta['scene'],
            'variant': meta['variant'],
            'frame_idx': 0,
            'camera': CameraParams(fx, fy, cx, cy, R, t, w, h),
        }
        
        return sample
    
    def get_frames(self, idx: int, num_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取多帧"""
        meta = self.samples[idx]
        
        frame_files = sorted(meta['image_dir'].glob('*.jpg'))
        
        if num_frames is not None:
            total = len(frame_files)
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        depth_dir = meta['scene_dir'] / 'frames' / 'depth' / 'Camera_0'
        
        frames = []
        
        for frame_file in frame_files:
            image = Image.open(frame_file).convert('RGB')
            
            # 加载深度
            depth_file = depth_dir / (frame_file.stem + '.png')
            if depth_file.exists():
                depth = np.array(Image.open(depth_file), dtype=np.float32)
                depth = depth / 100.0
            else:
                depth = None
            
            w, h = image.size
            fx = fy = 725.0
            cx = w / 2
            cy = h / 2
            R = np.eye(3)
            t = np.zeros(3)
            
            frames.append({
                'image': image,
                'depth': depth,
                'frame_idx': int(frame_file.stem.split('_')[-1]),
                'camera': CameraParams(fx, fy, cx, cy, R, t, w, h),
            })
        
        return frames
    
    def get_frame_timestamps(self, idx: int) -> List[float]:
        """获取时间戳"""
        meta = self.samples[idx]
        return list(range(meta['num_frames']))


# ============================================================================
# Builder Functions
# ============================================================================

def build_dataset(cfg, split: str = 'train', **kwargs) -> BaseDataset:
    """构建数据集
    
    Args:
        cfg: 配置对象
        split: 数据划分
        **kwargs: 额外参数
        
    Returns:
        数据集实例
    """
    defaults = OmegaConf.create({
        "data": {
            "root": "./data",
            "cache": {"enabled": False, "cache_dir": None},
            "shuffle": True,
            "pin_memory": False,
        },
        "num_workers": 0,
    })
    cfg = OmegaConf.merge(defaults, cfg)

    dataset_name = cfg.data.name.lower()
    
    # 构建变换
    transform = kwargs.pop('transform', None)
    if transform is None and hasattr(cfg.data, 'image'):
        transform = get_default_transforms(cfg.data, split)
    
    # 通用参数
    common_args = {
        'root': cfg.data.root,
        'split': split,
        'transform': transform,
        'cache_dir': cfg.data.cache.cache_dir if cfg.data.cache.enabled else None,
        'use_cache': cfg.data.cache.enabled,
    }
    common_args.update(kwargs)
    
    # 根据名称构建
    if dataset_name == 'pix3d':
        return Pix3DDataset(**common_args)
    
    elif dataset_name == 'co3dv2' or dataset_name == 'co3d':
        return CO3Dv2Dataset(**common_args)
    
    elif dataset_name == 'scannet':
        return ScanNetDataset(**common_args)
    
    elif dataset_name == 'kitti':
        return KITTIDataset(**common_args)
    
    elif dataset_name == 'vkitti' or dataset_name == 'virtual_kitti':
        return VirtualKITTIDataset(**common_args)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_dataloader(
    cfg,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """构建数据加载器
    
    Args:
        cfg: 配置对象
        split: 数据划分
        **kwargs: 额外参数
        
    Returns:
        DataLoader实例
    """
    defaults = OmegaConf.create({
        "data": {
            "shuffle": True,
            "pin_memory": False,
            "batch_size": 1,
        },
        "num_workers": 0,
    })
    cfg = OmegaConf.merge(defaults, cfg)

    # 构建数据集
    dataset = build_dataset(cfg, split=split)

    # DataLoader参数
    batch_size = kwargs.pop('batch_size', cfg.data.batch_size)
    shuffle = kwargs.pop('shuffle', split == 'train' and cfg.data.shuffle)
    num_workers = kwargs.pop('num_workers', cfg.num_workers)
    pin_memory = kwargs.pop('pin_memory', cfg.data.pin_memory)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
    
    return loader