"""几何工具函数

提供各种3D几何变换和计算功能。
"""

from typing import Tuple, Optional
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor
) -> torch.Tensor:
    """应用变换矩阵到点
    
    Args:
        points: (N, 3) 点坐标
        transform: (4, 4) 变换矩阵
        
    Returns:
        (N, 3) 变换后的点
    """
    # 齐次坐标
    ones = torch.ones(points.shape[0], 1, device=points.device)
    points_homo = torch.cat([points, ones], dim=1)  # (N, 4)
    
    # 变换
    transformed = points_homo @ transform.T  # (N, 4)
    
    # 转回3D坐标
    return transformed[:, :3]


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵转四元数
    
    Args:
        R: (3, 3) 或 (B, 3, 3) 旋转矩阵
        
    Returns:
        (4,) 或 (B, 4) 四元数 [w, x, y, z]
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    batch_size = R.shape[0]
    
    # 提取元素
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    
    # 计算四元数
    trace = r00 + r11 + r22
    
    quat = torch.zeros(batch_size, 4, device=R.device)
    
    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    quat[mask, 0] = 0.25 * s
    quat[mask, 1] = (r21[mask] - r12[mask]) / s
    quat[mask, 2] = (r02[mask] - r20[mask]) / s
    quat[mask, 3] = (r10[mask] - r01[mask]) / s
    
    # Case 2: r00 > r11 and r00 > r22
    mask = (~mask) & (r00 > r11) & (r00 > r22)
    s = torch.sqrt(1.0 + r00[mask] - r11[mask] - r22[mask]) * 2
    quat[mask, 0] = (r21[mask] - r12[mask]) / s
    quat[mask, 1] = 0.25 * s
    quat[mask, 2] = (r01[mask] + r10[mask]) / s
    quat[mask, 3] = (r02[mask] + r20[mask]) / s
    
    # Case 3: r11 > r22
    mask = (~mask) & (r11 > r22)
    s = torch.sqrt(1.0 + r11[mask] - r00[mask] - r22[mask]) * 2
    quat[mask, 0] = (r02[mask] - r20[mask]) / s
    quat[mask, 1] = (r01[mask] + r10[mask]) / s
    quat[mask, 2] = 0.25 * s
    quat[mask, 3] = (r12[mask] + r21[mask]) / s
    
    # Case 4: else
    mask = (~mask)
    s = torch.sqrt(1.0 + r22[mask] - r00[mask] - r11[mask]) * 2
    quat[mask, 0] = (r10[mask] - r01[mask]) / s
    quat[mask, 1] = (r02[mask] + r20[mask]) / s
    quat[mask, 2] = (r12[mask] + r21[mask]) / s
    quat[mask, 3] = 0.25 * s
    
    if squeeze:
        quat = quat.squeeze(0)
    
    return quat


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """四元数转旋转矩阵
    
    Args:
        q: (4,) 或 (B, 4) 四元数 [w, x, y, z]
        
    Returns:
        (3, 3) 或 (B, 3, 3) 旋转矩阵
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # 归一化
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # 构建旋转矩阵
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1),
    ], dim=1)
    
    if squeeze:
        R = R.squeeze(0)
    
    return R


def look_at(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """计算look-at变换矩阵
    
    Args:
        eye: (3,) 相机位置
        center: (3,) 观察目标位置
        up: (3,) 上方向
        
    Returns:
        (4, 4) 视图矩阵
    """
    # 方向向量
    f = center - eye
    f = f / torch.norm(f)
    
    # 右向量
    s = torch.cross(f, up)
    s = s / torch.norm(s)
    
    # 上向量
    u = torch.cross(s, f)
    
    # 构建矩阵
    M = torch.eye(4, device=eye.device)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -torch.stack([s.dot(eye), u.dot(eye), (-f).dot(eye)])
    
    return M


def perspective_projection(
    fov_y: float,
    aspect: float,
    near: float,
    far: float,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """计算透视投影矩阵
    
    Args:
        fov_y: 垂直视场角（弧度）
        aspect: 宽高比
        near: 近裁剪面
        far: 远裁剪面
        device: 设备
        
    Returns:
        (4, 4) 投影矩阵
    """
    f = 1.0 / np.tan(fov_y / 2.0)
    
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1.0
    
    return P


def compute_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor
) -> torch.Tensor:
    """计算网格顶点法线
    
    Args:
        vertices: (V, 3) 顶点坐标
        faces: (F, 3) 面索引
        
    Returns:
        (V, 3) 顶点法线
    """
    # 获取每个面的顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # 计算面法线
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = torch.cross(e1, e2, dim=1)
    face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
    
    # 累积到顶点
    vertex_normals = torch.zeros_like(vertices)
    
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_normals)
    
    # 归一化
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
    
    return vertex_normals


def sample_points_from_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从网格表面采样点
    
    Args:
        vertices: (V, 3) 顶点
        faces: (F, 3) 面索引
        num_samples: 采样点数
        
    Returns:
        (points, normals): 采样的点和法线
    """
    # 计算每个面的面积
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=1), dim=1)
    
    # 按面积加权采样面
    probs = areas / areas.sum()
    face_indices = torch.multinomial(probs, num_samples, replacement=True)
    
    # 在每个面上均匀采样
    r1 = torch.rand(num_samples, device=vertices.device)
    r2 = torch.rand(num_samples, device=vertices.device)
    
    # 重心坐标
    sqrt_r1 = torch.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2
    
    # 采样点
    v0_sampled = vertices[faces[face_indices, 0]]
    v1_sampled = vertices[faces[face_indices, 1]]
    v2_sampled = vertices[faces[face_indices, 2]]
    
    points = u.unsqueeze(1) * v0_sampled + \
             v.unsqueeze(1) * v1_sampled + \
             w.unsqueeze(1) * v2_sampled
    
    # 计算法线（面法线）
    e1_sampled = v1_sampled - v0_sampled
    e2_sampled = v2_sampled - v0_sampled
    normals = torch.cross(e1_sampled, e2_sampled, dim=1)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    return points, normals


def compute_bounding_box(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算点云的边界框
    
    Args:
        points: (N, 3) 点云
        
    Returns:
        (min_bound, max_bound): 最小和最大边界
    """
    min_bound = points.min(dim=0)[0]
    max_bound = points.max(dim=0)[0]
    return min_bound, max_bound


def normalize_points(
    points: torch.Tensor,
    center: bool = True,
    scale: bool = True
) -> Tuple[torch.Tensor, dict]:
    """归一化点云
    
    Args:
        points: (N, 3) 点云
        center: 是否中心化
        scale: 是否缩放到单位球
        
    Returns:
        (normalized_points, transform_params): 归一化的点云和变换参数
    """
    transform_params = {}
    
    if center:
        centroid = points.mean(dim=0)
        points = points - centroid
        transform_params['centroid'] = centroid
    
    if scale:
        max_dist = torch.norm(points, dim=1).max()
        points = points / max_dist
        transform_params['scale'] = max_dist
    
    return points, transform_params


def compute_rotation_from_vectors(
    v1: torch.Tensor,
    v2: torch.Tensor
) -> torch.Tensor:
    """计算从v1旋转到v2的旋转矩阵
    
    Args:
        v1: (3,) 源向量
        v2: (3,) 目标向量
        
    Returns:
        (3, 3) 旋转矩阵
    """
    # 归一化
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)
    
    # 旋转轴
    axis = torch.cross(v1, v2)
    axis_norm = torch.norm(axis)
    
    if axis_norm < 1e-6:
        # 向量平行
        return torch.eye(3, device=v1.device)
    
    axis = axis / axis_norm
    
    # 旋转角
    angle = torch.acos(torch.clamp(v1.dot(v2), -1.0, 1.0))
    
    # Rodrigues公式
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=v1.device)
    
    R = torch.eye(3, device=v1.device) + \
        torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)
    
    return R


def apply_transform_batch(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """批量应用刚体变换
    
    Args:
        points: (B, N, 3) 点云
        rotation: (B, 3, 3) 旋转矩阵
        translation: (B, 3) 平移向量
        scale: (B, 3) 可选的缩放
        
    Returns:
        (B, N, 3) 变换后的点云
    """
    # 旋转
    points_transformed = torch.bmm(points, rotation.transpose(1, 2))
    
    # 缩放
    if scale is not None:
        points_transformed = points_transformed * scale.unsqueeze(1)
    
    # 平移
    points_transformed = points_transformed + translation.unsqueeze(1)
    
    return points_transformed


def rodrigues_rotation(
    points: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor
) -> torch.Tensor:
    """Rodrigues旋转公式
    
    Args:
        points: (N, 3) 点
        axis: (3,) 旋转轴
        angle: 标量 旋转角（弧度）
        
    Returns:
        (N, 3) 旋转后的点
    """
    axis = axis / torch.norm(axis)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Rodrigues公式
    rotated = (
        points * cos_angle +
        torch.cross(axis.expand_as(points), points) * sin_angle +
        axis.unsqueeze(0) * (axis.unsqueeze(0) * points).sum(dim=1, keepdim=True) * (1 - cos_angle)
    )
    
    return rotated


def project_points_to_image(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将3D点投影到图像平面
    
    Args:
        points_3d: (N, 3) 3D点
        K: (3, 3) 相机内参
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
        image_size: (width, height)
        
    Returns:
        (points_2d, valid_mask): 2D点坐标和有效掩码
    """
    # 世界坐标到相机坐标
    points_cam = points_3d @ R.T + t
    
    # 投影到图像平面
    points_proj = points_cam @ K.T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    
    # 检查是否在图像内
    width, height = image_size
    valid_mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) &
        (points_cam[:, 2] > 0)  # 在相机前方
    )
    
    return points_2d, valid_mask