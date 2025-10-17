"""工具模块

提供几何计算、可视化、IO、日志等工具函数。
"""

from .geometry import (
    transform_points,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    look_at,
    perspective_projection,
    compute_normals,
    sample_points_from_mesh,
)

from .metrics import (
    compute_chamfer_distance,
    compute_iou,
    compute_fscore,
    compute_psnr,
    compute_ssim,
    evaluate_reconstruction,
)

from .visualization import (
    visualize_pointcloud,
    visualize_mesh,
    render_gaussian,
    plot_loss_curves,
    save_comparison_image,
)

from .io import (
    save_pointcloud,
    load_pointcloud,
    save_mesh,
    load_mesh,
    save_gaussian_model,
    load_gaussian_model,
)

from .logger import (
    setup_logger,
    get_logger,
    log_metrics,
    WandbLogger,
)

__all__ = [
    # Geometry
    'transform_points',
    'rotation_matrix_to_quaternion',
    'quaternion_to_rotation_matrix',
    'look_at',
    'perspective_projection',
    'compute_normals',
    'sample_points_from_mesh',
    # Metrics
    'compute_chamfer_distance',
    'compute_iou',
    'compute_fscore',
    'compute_psnr',
    'compute_ssim',
    'evaluate_reconstruction',
    # Visualization
    'visualize_pointcloud',
    'visualize_mesh',
    'render_gaussian',
    'plot_loss_curves',
    'save_comparison_image',
    # IO
    'save_pointcloud',
    'load_pointcloud',
    'save_mesh',
    'load_mesh',
    'save_gaussian_model',
    'load_gaussian_model',
    # Logger
    'setup_logger',
    'get_logger',
    'log_metrics',
    'WandbLogger',
]