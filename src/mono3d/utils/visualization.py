"""可视化工具

提供点云、网格、渲染结果的可视化功能。
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def visualize_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    point_size: float = 2.0,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Point Cloud"
) -> Optional[plt.Figure]:
    """可视化点云
    
    Args:
        points: (N, 3) 点坐标
        colors: (N, 3) 可选的颜色 [0, 1]
        normals: (N, 3) 可选的法线
        point_size: 点大小
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象（如果不显示）
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        log.error("matplotlib 3D toolkit not available")
        return None
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    if colors is not None:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            s=point_size,
            alpha=0.6
        )
    else:
        # 使用z坐标着色
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=points[:, 2],
            cmap='viridis',
            s=point_size,
            alpha=0.6
        )
    
    # 绘制法线
    if normals is not None:
        # 降采样以避免太密集
        sample_indices = np.random.choice(
            len(points),
            min(100, len(points)),
            replace=False
        )
        ax.quiver(
            points[sample_indices, 0],
            points[sample_indices, 1],
            points[sample_indices, 2],
            normals[sample_indices, 0],
            normals[sample_indices, 1],
            normals[sample_indices, 2],
            length=0.1,
            color='r',
            alpha=0.5
        )
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置相等的坐标轴比例
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def visualize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Mesh"
) -> Optional[plt.Figure]:
    """可视化网格
    
    Args:
        vertices: (V, 3) 顶点
        faces: (F, 3) 面索引
        vertex_colors: (V, 3) 可选的顶点颜色
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        log.error("matplotlib 3D toolkit not available")
        return None
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建三角形集合
    triangles = vertices[faces]
    
    if vertex_colors is not None:
        # 使用顶点颜色
        face_colors = vertex_colors[faces].mean(axis=1)
        mesh = Poly3DCollection(triangles, facecolors=face_colors, alpha=0.7, edgecolor='k', linewidths=0.1)
    else:
        mesh = Poly3DCollection(triangles, alpha=0.7, edgecolor='k', linewidths=0.1)
    
    ax.add_collection3d(mesh)
    
    # 设置坐标轴范围
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def render_gaussian(
    gaussian_model,
    camera_params: Dict[str, Any],
    save_path: Optional[Path] = None
) -> np.ndarray:
    """渲染高斯模型
    
    Args:
        gaussian_model: GaussianModel实例
        camera_params: 相机参数
        save_path: 保存路径
        
    Returns:
        渲染的图像 (H, W, 3)
    """
    with torch.no_grad():
        rendered = gaussian_model.render(camera_params)
        
        # 转换为numpy
        image = rendered['color'].cpu().permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
    
    if save_path:
        plt.imsave(save_path, image)
        log.info(f"Saved rendered image to {save_path}")
    
    return image


def plot_loss_curves(
    losses: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Training Loss"
) -> plt.Figure:
    """绘制损失曲线
    
    Args:
        losses: 损失字典，键为损失名称，值为损失历史
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象
    """
    fig, axes = plt.subplots(
        len(losses),
        1,
        figsize=(10, 3 * len(losses)),
        squeeze=False
    )
    
    for idx, (name, values) in enumerate(losses.items()):
        ax = axes[idx, 0]
        ax.plot(values, label=name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved loss curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def save_comparison_image(
    images: List[np.ndarray],
    labels: List[str],
    save_path: Path,
    figsize: Tuple[int, int] = (15, 5)
):
    """保存对比图像
    
    Args:
        images: 图像列表
        labels: 标签列表
        save_path: 保存路径
        figsize: 图像大小
    """
    num_images = len(images)
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    if num_images == 1:
        axes = [axes]
    
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved comparison image to {save_path}")


def visualize_depth(
    depth: np.ndarray,
    cmap: str = 'turbo',
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Depth Map"
) -> Optional[plt.Figure]:
    """可视化深度图
    
    Args:
        depth: (H, W) 深度图
        cmap: 颜色映射
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(depth, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved depth visualization to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def visualize_attention_map(
    attention: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Attention Map"
) -> Optional[plt.Figure]:
    """可视化注意力图
    
    Args:
        attention: (H, W) 注意力权重
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved attention map to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def create_video_from_images(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 30,
    codec: str = 'mp4v'
):
    """从图像序列创建视频
    
    Args:
        image_paths: 图像路径列表
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码
    """
    try:
        import cv2
    except ImportError:
        log.error("OpenCV not installed, cannot create video")
        return
    
    # 读取第一张图像以获取尺寸
    first_img = cv2.imread(str(image_paths[0]))
    height, width = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 写入所有帧
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        writer.write(img)
    
    writer.release()
    log.info(f"Created video at {output_path}")


def plot_3d_trajectory(
    positions: np.ndarray,
    orientations: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Camera Trajectory"
) -> Optional[plt.Figure]:
    """绘制3D轨迹
    
    Args:
        positions: (N, 3) 位置序列
        orientations: (N, 3, 3) 可选的旋转矩阵序列
        save_path: 保存路径
        show: 是否显示
        title: 标题
        
    Returns:
        Figure对象
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        log.error("matplotlib 3D toolkit not available")
        return None
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # 绘制朝向
    if orientations is not None:
        # 降采样
        step = max(1, len(positions) // 20)
        for i in range(0, len(positions), step):
            pos = positions[i]
            rot = orientations[i]
            
            # 绘制坐标轴
            axis_length = 0.5
            for j, color in enumerate(['r', 'g', 'b']):
                direction = rot[:, j] * axis_length
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    direction[0], direction[1], direction[2],
                    color=color,
                    alpha=0.6,
                    arrow_length_ratio=0.3
                )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved trajectory plot to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig


def visualize_multi_view(
    images: List[np.ndarray],
    cameras: List[Dict[str, Any]],
    point_cloud: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """可视化多视角场景
    
    Args:
        images: 图像列表
        cameras: 相机参数列表
        point_cloud: 可选的点云
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        Figure对象
    """
    num_views = len(images)
    cols = min(4, num_views)
    rows = (num_views + cols - 1) // cols
    
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f'View {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved multi-view visualization to {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        plt.close()
        return fig