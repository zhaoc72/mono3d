#!/usr/bin/env python3
"""
诊断脚本 - 测试单张图像处理
"""
import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

print("=" * 80)
print("单张图像处理诊断")
print("=" * 80)

# 配置路径
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
config_path = "configs/model_config.yaml"
prompt_config_path = "configs/prompt_config.yaml"
output_dir = Path("outputs/single_image_test")

print(f"\n1. 检查文件是否存在...")
print(f"   图像: {image_path}")
print(f"   存在: {Path(image_path).exists()}")
print(f"   配置: {config_path}")
print(f"   存在: {Path(config_path).exists()}")
print(f"   Prompt配置: {prompt_config_path}")
print(f"   存在: {Path(prompt_config_path).exists()}")

if not Path(image_path).exists():
    print(f"\n❌ 图像文件不存在: {image_path}")
    sys.exit(1)

if not Path(config_path).exists():
    print(f"\n❌ 配置文件不存在: {config_path}")
    sys.exit(1)

try:
    print(f"\n2. 加载图像...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ❌ 无法读取图像")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   ✓ 图像形状: {image.shape}")
    
    print(f"\n3. 加载配置...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ 主配置加载成功")
    print(f"   设备: {config.get('device', 'N/A')}")
    print(f"   精度: {config.get('dtype', 'N/A')}")
    
    if Path(prompt_config_path).exists():
        with open(prompt_config_path, 'r') as f:
            prompt_config = yaml.safe_load(f)
        print(f"   ✓ Prompt配置加载成功")
    else:
        prompt_config = {}
        print(f"   ⚠️  Prompt配置文件不存在，使用默认配置")
    
    print(f"\n4. 检查模型路径...")
    dinov3_path = config['dinov3']['checkpoint_path']
    sam2_path = config['sam2']['checkpoint_path']
    
    print(f"   DINOv3: {dinov3_path}")
    print(f"   存在: {Path(dinov3_path).exists()}")
    
    print(f"   SAM2: {sam2_path}")
    print(f"   存在: {Path(sam2_path).exists()}")
    
    if not Path(dinov3_path).exists():
        print(f"\n   ❌ DINOv3 checkpoint 不存在")
        sys.exit(1)
    
    if not Path(sam2_path).exists():
        print(f"\n   ❌ SAM2 checkpoint 不存在")
        sys.exit(1)
    
    print(f"\n5. 检查 CUDA...")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\n6. 导入模块...")
    try:
        from src.data_loader import load_image
        from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
        from src.sam2_segmenter import SAM2Segmenter, Sam2Config
        from src.inference_pipeline import ZeroShotSegmentationPipeline, PipelineConfig
        from src.prompt_generator import ClusterConfig, PromptConfig
        from src.utils import to_torch_dtype, ensure_directory
        print(f"   ✓ 所有模块导入成功")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        sys.exit(1)
    
    print(f"\n7. 初始化 DINOv3...")
    device = torch.device(config['device'])
    dtype = to_torch_dtype(config['dtype'])
    
    dinov3_cfg = Dinov3Config(**config['dinov3'])
    try:
        extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
        print(f"   ✓ DINOv3 初始化成功")
    except Exception as e:
        print(f"   ❌ DINOv3 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n8. 初始化 SAM2...")
    sam2_cfg = Sam2Config(**config['sam2'])
    try:
        segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
        print(f"   ✓ SAM2 初始化成功")
    except Exception as e:
        print(f"   ❌ SAM2 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n9. 初始化 Pipeline...")
    pipeline_dict = config.get('pipeline', {})
    cluster_cfg = ClusterConfig(**pipeline_dict.get('cluster', {}))
    
    # PromptConfig 只支持这些参数
    # prompt_config.yaml 中的其他参数（如 normalize, threshold_strategy 等）
    # 是用于其他地方的，不是 PromptConfig 的参数
    prompt_dict = pipeline_dict.get('prompt', {})
    
    # 只使用 PromptConfig 真正支持的参数
    prompt_cfg = PromptConfig(
        include_boxes=prompt_dict.get('include_boxes', True),
        include_points=prompt_dict.get('include_points', True),
        point_strategy=prompt_dict.get('point_strategy', 'density'),
        max_points_per_region=prompt_dict.get('max_points_per_region', 5),
        density_noise_handling=prompt_dict.get('density_noise_handling', 'nearest')
    )
    
    pipeline_cfg = PipelineConfig(
        cluster=cluster_cfg,
        prompt=prompt_cfg
    )
    
    try:
        pipeline = ZeroShotSegmentationPipeline(
            dinov3_cfg,
            sam2_cfg,
            pipeline_cfg,
            device=str(device),
            dtype=dtype,
            extractor=extractor,
            segmenter=segmenter
        )
        print(f"   ✓ Pipeline 初始化成功")
    except Exception as e:
        print(f"   ❌ Pipeline 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n10. 运行推理...")
    nms_config = {
        "enable_nms": pipeline_dict.get("enable_nms", True),
        "iou_threshold": pipeline_dict.get("iou_threshold", 0.6),
        "objectness_weight": pipeline_dict.get("objectness_weight", 0.5),
        "confidence_weight": pipeline_dict.get("confidence_weight", 0.3),
        "area_weight": pipeline_dict.get("area_weight", 0.2),
    }
    
    try:
        result = pipeline.run(image, nms_config=nms_config)
        print(f"   ✓ 推理完成")
        print(f"   候选区域: {len(result.proposals)}")
        print(f"   生成掩码: {len(result.masks)}")
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n11. 保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存掩码
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    
    area_threshold = pipeline_dict.get("area_threshold", 100)
    valid_count = 0
    
    for idx, mask in enumerate(result.masks):
        area = int(mask.astype(np.uint8).sum())
        if area >= area_threshold:
            mask_path = mask_dir / f"mask_{idx:03d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            valid_count += 1
    
    print(f"   ✓ 保存了 {valid_count} 个掩码")
    
    # 保存可视化
    if result.masks:
        combined = np.zeros_like(image)
        for idx, mask in enumerate(result.masks):
            color = np.array([
                (idx * 50) % 255,
                (idx * 80 + 60) % 255,
                (idx * 120 + 30) % 255
            ], dtype=np.uint8)
            combined[mask.astype(bool)] = color
        
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        
        viz_path = output_dir / "visualization.png"
        cv2.imwrite(str(viz_path), overlay)
        print(f"   ✓ 可视化已保存: {viz_path}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ 测试成功完成！")
    print(f"输出目录: {output_dir}")
    print(f"=" * 80)
    
except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)