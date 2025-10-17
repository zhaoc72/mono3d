"""IO工具

提供文件读写功能，支持点云、网格、模型保存和加载。
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch
from pathlib import Path
import struct
import logging

log = logging.getLogger(__name__)


def save_pointcloud(
    points: np.ndarray,
    filepath: Path,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    format: str = 'auto'
):
    """保存点云
    
    Args:
        points: (N, 3) 点坐标
        filepath: 文件路径
        colors: (N, 3) 可选的颜色 [0, 1]
        normals: (N, 3) 可选的法线
        format: 文件格式 ('ply', 'xyz', 'auto')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = filepath.suffix[1:]  # 去掉'.'
    
    if format == 'ply':
        _save_ply(filepath, points, colors, normals)
    elif format == 'xyz':
        _save_xyz(filepath, points, colors)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    log.info(f"Saved point cloud to {filepath}")


def _save_ply(
    filepath: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray]
):
    """保存PLY格式"""
    num_points = len(points)
    
    # 写入header
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # 写入数据
        for i in range(num_points):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            
            if normals is not None:
                line += f" {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}"
            
            if colors is not None:
                r, g, b = (colors[i] * 255).astype(int)
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")


def _save_xyz(filepath: Path, points: np.ndarray, colors: Optional[np.ndarray]):
    """保存XYZ格式"""
    with open(filepath, 'w') as f:
        for i in range(len(points)):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            
            if colors is not None:
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            
            f.write(line + "\n")


def load_pointcloud(filepath: Path, format: str = 'auto') -> Dict[str, np.ndarray]:
    """加载点云
    
    Args:
        filepath: 文件路径
        format: 文件格式
        
    Returns:
        点云字典，包含 'points', 'colors', 'normals' 等
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")
    
    if format == 'auto':
        format = filepath.suffix[1:]
    
    if format == 'ply':
        return _load_ply(filepath)
    elif format == 'xyz':
        return _load_xyz(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_ply(filepath: Path) -> Dict[str, np.ndarray]:
    """加载PLY格式"""
    with open(filepath, 'r') as f:
        # 读取header
        line = f.readline()
        assert line.strip() == 'ply'
        
        num_vertices = 0
        has_normals = False
        has_colors = False
        
        while True:
            line = f.readline().strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property') and 'nx' in line:
                has_normals = True
            elif line.startswith('property') and 'red' in line:
                has_colors = True
            elif line == 'end_header':
                break
        
        # 读取数据
        points = []
        colors = [] if has_colors else None
        normals = [] if has_normals else None
        
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            idx = 0
            
            # 坐标
            points.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
            idx += 3
            
            # 法线
            if has_normals:
                normals.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
                idx += 3
            
            # 颜色
            if has_colors:
                colors.append([int(parts[idx]) / 255.0, int(parts[idx+1]) / 255.0, int(parts[idx+2]) / 255.0])
    
    result = {'points': np.array(points, dtype=np.float32)}
    
    if colors is not None:
        result['colors'] = np.array(colors, dtype=np.float32)
    
    if normals is not None:
        result['normals'] = np.array(normals, dtype=np.float32)
    
    log.info(f"Loaded point cloud from {filepath}")
    
    return result


def _load_xyz(filepath: Path) -> Dict[str, np.ndarray]:
    """加载XYZ格式"""
    data = np.loadtxt(filepath)
    
    result = {'points': data[:, :3].astype(np.float32)}
    
    if data.shape[1] >= 6:
        result['colors'] = data[:, 3:6].astype(np.float32)
    
    log.info(f"Loaded point cloud from {filepath}")
    
    return result


def save_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: Path,
    vertex_colors: Optional[np.ndarray] = None,
    vertex_normals: Optional[np.ndarray] = None,
    format: str = 'auto'
):
    """保存网格
    
    Args:
        vertices: (V, 3) 顶点
        faces: (F, 3) 面索引
        filepath: 文件路径
        vertex_colors: (V, 3) 可选的顶点颜色
        vertex_normals: (V, 3) 可选的顶点法线
        format: 文件格式
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = filepath.suffix[1:]
    
    if format == 'obj':
        _save_obj(filepath, vertices, faces, vertex_colors, vertex_normals)
    elif format == 'ply':
        _save_mesh_ply(filepath, vertices, faces, vertex_colors, vertex_normals)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    log.info(f"Saved mesh to {filepath}")


def _save_obj(
    filepath: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray],
    vertex_normals: Optional[np.ndarray]
):
    """保存OBJ格式"""
    with open(filepath, 'w') as f:
        # 写入顶点
        for i, v in enumerate(vertices):
            line = f"v {v[0]} {v[1]} {v[2]}"
            
            if vertex_colors is not None:
                r, g, b = vertex_colors[i]
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")
        
        # 写入法线
        if vertex_normals is not None:
            for n in vertex_normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        # 写入面
        for face in faces:
            if vertex_normals is not None:
                # 带法线
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _save_mesh_ply(
    filepath: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray],
    vertex_normals: Optional[np.ndarray]
):
    """保存PLY网格"""
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if vertex_normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        
        if vertex_colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # 顶点
        for i, v in enumerate(vertices):
            line = f"{v[0]} {v[1]} {v[2]}"
            
            if vertex_normals is not None:
                line += f" {vertex_normals[i, 0]} {vertex_normals[i, 1]} {vertex_normals[i, 2]}"
            
            if vertex_colors is not None:
                r, g, b = (vertex_colors[i] * 255).astype(int)
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")
        
        # 面
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load_mesh(filepath: Path, format: str = 'auto') -> Dict[str, np.ndarray]:
    """加载网格
    
    Args:
        filepath: 文件路径
        format: 文件格式
        
    Returns:
        网格字典
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    
    if format == 'auto':
        format = filepath.suffix[1:]
    
    if format == 'obj':
        return _load_obj(filepath)
    elif format == 'ply':
        return _load_mesh_ply(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_obj(filepath: Path) -> Dict[str, np.ndarray]:
    """加载OBJ格式"""
    vertices = []
    faces = []
    normals = []
    vertex_colors = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            if parts[0] == 'v':
                # 顶点
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                
                # 可选的颜色
                if len(parts) >= 7:
                    color = [float(parts[4]), float(parts[5]), float(parts[6])]
                    vertex_colors.append(color)
            
            elif parts[0] == 'vn':
                # 法线
                normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                normals.append(normal)
            
            elif parts[0] == 'f':
                # 面
                face = []
                for part in parts[1:]:
                    # 处理 v/vt/vn 格式
                    indices = part.split('/')
                    v_idx = int(indices[0]) - 1  # OBJ索引从1开始
                    face.append(v_idx)
                
                # 只支持三角形
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    # 四边形转两个三角形
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    
    result = {
        'vertices': np.array(vertices, dtype=np.float32),
        'faces': np.array(faces, dtype=np.int32),
    }
    
    if vertex_colors:
        result['vertex_colors'] = np.array(vertex_colors, dtype=np.float32)
    
    if normals:
        result['normals'] = np.array(normals, dtype=np.float32)
    
    log.info(f"Loaded mesh from {filepath}")
    
    return result


def _load_mesh_ply(filepath: Path) -> Dict[str, np.ndarray]:
    """加载PLY网格"""
    # 简化实现，类似_load_ply但增加面的处理
    with open(filepath, 'r') as f:
        # 解析header
        line = f.readline()
        assert line.strip() == 'ply'
        
        num_vertices = 0
        num_faces = 0
        has_normals = False
        has_colors = False
        
        while True:
            line = f.readline().strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line.startswith('property') and 'nx' in line:
                has_normals = True
            elif line.startswith('property') and 'red' in line:
                has_colors = True
            elif line == 'end_header':
                break
        
        # 读取顶点
        vertices = []
        colors = [] if has_colors else None
        normals = [] if has_normals else None
        
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            idx = 0
            
            vertices.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
            idx += 3
            
            if has_normals:
                normals.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
                idx += 3
            
            if has_colors:
                colors.append([int(parts[idx]) / 255.0, int(parts[idx+1]) / 255.0, int(parts[idx+2]) / 255.0])
        
        # 读取面
        faces = []
        for _ in range(num_faces):
            parts = f.readline().strip().split()
            num_verts = int(parts[0])
            face_verts = [int(parts[i+1]) for i in range(num_verts)]
            
            # 只支持三角形
            if num_verts == 3:
                faces.append(face_verts)
    
    result = {
        'vertices': np.array(vertices, dtype=np.float32),
        'faces': np.array(faces, dtype=np.int32),
    }
    
    if colors:
        result['vertex_colors'] = np.array(colors, dtype=np.float32)
    
    if normals:
        result['normals'] = np.array(normals, dtype=np.float32)
    
    log.info(f"Loaded mesh from {filepath}")
    
    return result


def save_gaussian_model(
    gaussian_model,
    filepath: Path
):
    """保存Gaussian模型
    
    Args:
        gaussian_model: GaussianModel实例
        filepath: 文件路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存参数
    state_dict = {
        'xyz': gaussian_model._xyz.data.cpu(),
        'features_dc': gaussian_model._features_dc.data.cpu(),
        'features_rest': gaussian_model._features_rest.data.cpu(),
        'opacity': gaussian_model._opacity.data.cpu(),
        'scaling': gaussian_model._scaling.data.cpu(),
        'rotation': gaussian_model._rotation.data.cpu(),
        'num_gaussians': gaussian_model.num_gaussians,
        'sh_degree': gaussian_model.sh_degree,
    }
    
    torch.save(state_dict, filepath)
    log.info(f"Saved Gaussian model to {filepath}")


def load_gaussian_model(
    filepath: Path,
    device: torch.device = torch.device('cpu')
):
    """加载Gaussian模型
    
    Args:
        filepath: 文件路径
        device: 设备
        
    Returns:
        GaussianModel实例
    """
    from ..models.gaussian import GaussianModel
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Gaussian model file not found: {filepath}")
    
    # 加载参数
    state_dict = torch.load(filepath, map_location=device)
    
    # 创建模型
    model = GaussianModel(
        num_gaussians=state_dict['num_gaussians'],
        sh_degree=state_dict['sh_degree']
    )
    
    # 加载参数
    model._xyz.data = state_dict['xyz'].to(device)
    model._features_dc.data = state_dict['features_dc'].to(device)
    model._features_rest.data = state_dict['features_rest'].to(device)
    model._opacity.data = state_dict['opacity'].to(device)
    model._scaling.data = state_dict['scaling'].to(device)
    model._rotation.data = state_dict['rotation'].to(device)
    
    log.info(f"Loaded Gaussian model from {filepath}")
    
    return model


def save_config(config: Dict[str, Any], filepath: Path):
    """保存配置文件
    
    Args:
        config: 配置字典
        filepath: 文件路径
    """
    import json
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    log.info(f"Saved config to {filepath}")


def load_config(filepath: Path) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        配置字典
    """
    import json
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    log.info(f"Loaded config from {filepath}")
    
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path,
    **kwargs
):
    """保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 文件路径
        **kwargs: 其他要保存的内容
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 添加额外参数
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    log.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """加载训练检查点
    
    Args:
        filepath: 文件路径
        model: 可选的模型（加载权重）
        optimizer: 可选的优化器（加载状态）
        device: 设备
        
    Returns:
        检查点字典
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    log.info(f"Loaded checkpoint from {filepath}")
    
    return checkpoint


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filepath: Path,
    input_names: List[str] = ['input'],
    output_names: List[str] = ['output'],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
):
    """导出模型到ONNX格式
    
    Args:
        model: PyTorch模型
        dummy_input: 示例输入
        filepath: 输出路径
        input_names: 输入名称列表
        output_names: 输出名称列表
        dynamic_axes: 动态轴配置
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        str(filepath),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )
    
    log.info(f"Exported model to ONNX format: {filepath}")


def compress_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.01
) -> Dict[str, np.ndarray]:
    """体素下采样压缩点云
    
    Args:
        points: (N, 3) 点云
        colors: (N, 3) 可选的颜色
        voxel_size: 体素大小
        
    Returns:
        下采样后的点云
    """
    # 量化到体素网格
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # 找唯一体素
    unique_voxels, inverse_indices = np.unique(
        voxel_coords,
        axis=0,
        return_inverse=True
    )
    
    # 对每个体素内的点求平均
    compressed_points = []
    compressed_colors = [] if colors is not None else None
    
    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        compressed_points.append(points[mask].mean(axis=0))
        
        if colors is not None:
            compressed_colors.append(colors[mask].mean(axis=0))
    
    result = {'points': np.array(compressed_points, dtype=np.float32)}
    
    if compressed_colors is not None:
        result['colors'] = np.array(compressed_colors, dtype=np.float32)
    
    log.info(f"Compressed point cloud from {len(points)} to {len(compressed_points)} points")
    
    return result