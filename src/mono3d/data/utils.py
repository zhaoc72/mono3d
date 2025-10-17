"""数据工具函数

提供相机参数处理、视频处理、点云转换等工具。
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import json
import logging

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

log = logging.getLogger(__name__)


@dataclass
class CameraParams:
    """相机参数数据类"""
    
    # 内参
    fx: float  # 焦距 x
    fy: float  # 焦距 y
    cx: float  # 主点 x
    cy: float  # 主点 y
    
    # 外参 (世界坐标系到相机坐标系)
    R: np.ndarray  # 旋转矩阵 (3, 3)
    t: np.ndarray  # 平移向量 (3,)
    
    # 图像尺寸
    width: int
    height: int
    
    # 可选：畸变参数
    distortion: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'R': self.R.tolist(),
            't': self.t.tolist(),
            'width': self.width,
            'height': self.height,
            'distortion': self.distortion.tolist() if self.distortion is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraParams':
        """从字典创建"""
        return cls(
            fx=data['fx'],
            fy=data['fy'],
            cx=data['cx'],
            cy=data['cy'],
            R=np.array(data['R']),
            t=np.array(data['t']),
            width=data['width'],
            height=data['height'],
            distortion=np.array(data['distortion']) if data.get('distortion') else None,
        )
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """获取内参矩阵 K (3x3)"""
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def get_extrinsic_matrix(self) -> np.ndarray:
        """获取外参矩阵 [R|t] (3x4)"""
        return np.hstack([self.R, self.t.reshape(3, 1)])
    
    def get_projection_matrix(self) -> np.ndarray:
        """获取投影矩阵 P = K[R|t] (3x4)"""
        K = self.get_intrinsic_matrix()
        extrinsic = self.get_extrinsic_matrix()
        return K @ extrinsic
    
    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """世界坐标系到相机坐标系
        
        Args:
            points_world: (N, 3) 世界坐标点
            
        Returns:
            (N, 3) 相机坐标点
        """
        points_cam = points_world @ self.R.T + self.t
        return points_cam
    
    def camera_to_image(self, points_cam: np.ndarray) -> np.ndarray:
        """相机坐标系到图像坐标系
        
        Args:
            points_cam: (N, 3) 相机坐标点
            
        Returns:
            (N, 2) 图像坐标点
        """
        x = points_cam[:, 0] / points_cam[:, 2]
        y = points_cam[:, 1] / points_cam[:, 2]
        
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        
        return np.stack([u, v], axis=1)
    
    def project(self, points_world: np.ndarray) -> np.ndarray:
        """世界坐标投影到图像坐标
        
        Args:
            points_world: (N, 3) 世界坐标点
            
        Returns:
            (N, 2) 图像坐标点
        """
        points_cam = self.world_to_camera(points_world)
        points_img = self.camera_to_image(points_cam)
        return points_img


def read_camera_params(
    filepath: Path,
    format: str = 'auto'
) -> CameraParams:
    """读取相机参数文件
    
    Args:
        filepath: 文件路径
        format: 格式 ('auto', 'json', 'txt', 'colmap', 'scannet')
        
    Returns:
        CameraParams对象
    """
    filepath = Path(filepath)
    
    if format == 'auto':
        # 根据扩展名自动判断
        if filepath.suffix == '.json':
            format = 'json'
        elif filepath.suffix == '.txt':
            format = 'txt'
        else:
            raise ValueError(f"Cannot infer format from {filepath}")
    
    if format == 'json':
        with open(filepath) as f:
            data = json.load(f)
        return CameraParams.from_dict(data)
    
    elif format == 'txt':
        # 简单文本格式：每行一个参数
        with open(filepath) as f:
            lines = f.readlines()
        
        # 解析 (示例格式)
        fx = float(lines[0].split()[1])
        fy = float(lines[1].split()[1])
        cx = float(lines[2].split()[1])
        cy = float(lines[3].split()[1])
        
        R = np.array([
            [float(x) for x in lines[4].split()[1:]],
            [float(x) for x in lines[5].split()[1:]],
            [float(x) for x in lines[6].split()[1:]],
        ])
        
        t = np.array([float(x) for x in lines[7].split()[1:]])
        
        width = int(lines[8].split()[1])
        height = int(lines[9].split()[1])
        
        return CameraParams(fx, fy, cx, cy, R, t, width, height)
    
    elif format == 'colmap':
        # COLMAP格式
        # TODO: 实现COLMAP格式解析
        raise NotImplementedError("COLMAP format not implemented")
    
    elif format == 'scannet':
        # ScanNet格式
        # TODO: 实现ScanNet格式解析
        raise NotImplementedError("ScanNet format not implemented")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def save_camera_params(
    camera: CameraParams,
    filepath: Path,
    format: str = 'json'
):
    """保存相机参数
    
    Args:
        camera: 相机参数
        filepath: 保存路径
        format: 格式
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(camera.to_dict(), f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def extract_video_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    quality: int = 95,
) -> List[Path]:
    """提取视频帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录（None表示不保存）
        fps: 目标帧率（None表示原始帧率）
        max_frames: 最大帧数
        start_frame: 起始帧
        quality: JPEG质量 (0-100)
        
    Returns:
        提取的帧路径列表（如果保存）或帧数组列表
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for extract_video_frames but is not available")

    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(
        f"Video: {video_path.name}, "
        f"FPS: {original_fps:.2f}, "
        f"Total frames: {total_frames}"
    )
    
    # 计算采样间隔
    if fps is not None:
        frame_interval = int(original_fps / fps)
    else:
        frame_interval = 1
    
    # 创建输出目录
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    frame_idx = 0
    saved_idx = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 采样
        if frame_idx % frame_interval == 0:
            if output_dir is not None:
                # 保存到磁盘
                frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                extracted_frames.append(frame_path)
            else:
                # 保存到内存
                extracted_frames.append(frame)
            
            saved_idx += 1
            
            # 检查最大帧数
            if max_frames is not None and saved_idx >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    log.info(f"Extracted {len(extracted_frames)} frames")
    
    return extracted_frames


def depth_to_pointcloud(
    depth: np.ndarray,
    camera: CameraParams,
    mask: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """深度图转点云
    
    Args:
        depth: 深度图 (H, W)
        camera: 相机参数
        mask: 可选的掩码 (H, W)
        color: 可选的颜色图 (H, W, 3)
        
    Returns:
        点云字典:
            - points: (N, 3) 三维坐标
            - colors: (N, 3) 颜色 (如果提供)
    """
    h, w = depth.shape
    
    # 创建像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 应用掩码
    if mask is not None:
        valid = mask > 0
    else:
        valid = depth > 0
    
    u = u[valid]
    v = v[valid]
    z = depth[valid]
    
    # 反投影到相机坐标系
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    
    points_cam = np.stack([x, y, z], axis=1)
    
    # 转换到世界坐标系
    R_inv = camera.R.T
    t_inv = -R_inv @ camera.t
    points_world = points_cam @ R_inv.T + t_inv
    
    result = {'points': points_world}
    
    # 添加颜色
    if color is not None:
        if color.shape[:2] != (h, w):
            if cv2 is not None:
                color = cv2.resize(color, (w, h))
            else:  # pragma: no cover - executed only when OpenCV is unavailable
                from PIL import Image

                color = np.array(Image.fromarray(color).resize((w, h)))
        
        colors = color[valid]
        result['colors'] = colors / 255.0  # 归一化到 [0, 1]
    
    return result


def pointcloud_to_depth(
    points: np.ndarray,
    camera: CameraParams,
    colors: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """点云投影到深度图
    
    Args:
        points: (N, 3) 世界坐标点
        camera: 相机参数
        colors: (N, 3) 可选的颜色
        
    Returns:
        字典:
            - depth: (H, W) 深度图
            - color: (H, W, 3) 颜色图 (如果提供)
            - mask: (H, W) 有效像素掩码
    """
    # 投影到图像坐标
    points_cam = camera.world_to_camera(points)
    points_img = camera.camera_to_image(points_cam)
    
    # 深度值
    depths = points_cam[:, 2]
    
    # 创建深度图
    depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    
    if colors is not None:
        color_map = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
    
    # 遍历点
    for i, (u, v) in enumerate(points_img):
        u_int = int(round(u))
        v_int = int(round(v))
        
        # 检查边界
        if 0 <= u_int < camera.width and 0 <= v_int < camera.height:
            # 深度冲突：保留较近的点
            if not mask[v_int, u_int] or depths[i] < depth_map[v_int, u_int]:
                depth_map[v_int, u_int] = depths[i]
                mask[v_int, u_int] = True
                
                if colors is not None:
                    color_map[v_int, u_int] = colors[i]
    
    result = {
        'depth': depth_map,
        'mask': mask.astype(np.float32),
    }
    
    if colors is not None:
        result['color'] = color_map
    
    return result


def merge_pointclouds(
    pointclouds: List[Dict[str, np.ndarray]],
    voxel_size: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """合并多个点云
    
    Args:
        pointclouds: 点云列表，每个是包含'points'和可选'colors'的字典
        voxel_size: 体素下采样大小（None表示不下采样）
        
    Returns:
        合并后的点云
    """
    all_points = []
    all_colors = []
    has_color = 'colors' in pointclouds[0]
    
    for pc in pointclouds:
        all_points.append(pc['points'])
        if has_color:
            all_colors.append(pc['colors'])
    
    points = np.vstack(all_points)
    
    if has_color:
        colors = np.vstack(all_colors)
    
    # 体素下采样
    if voxel_size is not None:
        points, indices = voxel_downsample(points, voxel_size)
        if has_color:
            colors = colors[indices]
    
    result = {'points': points}
    if has_color:
        result['colors'] = colors
    
    return result


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """体素下采样
    
    Args:
        points: (N, 3) 点云
        voxel_size: 体素大小
        
    Returns:
        (downsampled_points, indices)
    """
    # 量化到体素网格
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # 找到唯一体素
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    
    return points[unique_indices], unique_indices


def compute_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """从掩码计算边界框
    
    Args:
        mask: (H, W) 二值掩码
        
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return int(x_min), int(y_min), int(x_max), int(y_max)


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """带填充的裁剪
    
    Args:
        image: 输入图像
        bbox: (x_min, y_min, x_max, y_max)
        padding: 填充比例
        
    Returns:
        (cropped_image, padded_bbox)
    """
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # 计算填充
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)
    
    # 应用填充并限制在图像边界内
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(w, x_max + pad_w)
    y_max = min(h, y_max + pad_h)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (x_min, y_min, x_max, y_max)


def estimate_camera_from_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    focal_length_factor: float = 1.5,
) -> CameraParams:
    """从边界框估计相机参数（用于单图像场景）
    
    Args:
        bbox: 物体边界框
        image_size: (width, height)
        focal_length_factor: 焦距因子
        
    Returns:
        估计的相机参数
    """
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox
    
    # 估计物体在图像中的尺寸
    obj_width = x_max - x_min
    obj_height = y_max - y_min
    obj_size = max(obj_width, obj_height)
    
    # 估计焦距（假设物体占据一定比例的视野）
    fx = fy = min(width, height) * focal_length_factor
    
    # 主点设为图像中心
    cx = width / 2
    cy = height / 2
    
    # 单位外参（相机在世界坐标系原点）
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    
    return CameraParams(fx, fy, cx, cy, R, t, width, height)