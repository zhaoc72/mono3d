"""验证 DINOv3 checkpoint 是否正确"""

import torch
from pathlib import Path
import sys

def verify_checkpoint(ckpt_path: str, checkpoint_type: str = "unknown"):
    """
    验证 checkpoint 文件
    
    Args:
        ckpt_path: checkpoint 文件路径
        checkpoint_type: 'backbone', 'detection', 'segmentation'
    """
    print("=" * 80)
    print(f"验证 Checkpoint: {ckpt_path}")
    print(f"类型: {checkpoint_type}")
    print("=" * 80)
    
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ 文件不存在: {ckpt_path}")
        return False
    
    # 检查文件大小
    file_size_gb = ckpt_path.stat().st_size / (1024**3)
    print(f"\n文件大小: {file_size_gb:.2f} GB")
    
    # 加载 checkpoint
    try:
        print("\n正在加载 checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("✓ 加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 分析结构
    print("\n" + "=" * 80)
    print("Checkpoint 结构分析")
    print("=" * 80)
    
    print(f"\n顶层键: {list(ckpt.keys())}")
    
    # 确定模型参数位置
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
            print("\n✓ 找到 'model' 键")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("\n✓ 找到 'state_dict' 键")
        else:
            state_dict = ckpt
            print("\n使用顶层字典作为 state_dict")
    else:
        print(f"\n❌ 未知的 checkpoint 格式: {type(ckpt)}")
        return False
    
    # 统计参数
    param_keys = list(state_dict.keys())
    print(f"\n参数总数: {len(param_keys)}")
    
    # 显示前 30 个参数
    print("\n前 30 个参数键:")
    for i, key in enumerate(param_keys[:30], 1):
        param = state_dict[key]
        if isinstance(param, torch.Tensor):
            print(f"  {i:3d}. {key:60s} | shape: {str(tuple(param.shape)):30s} | dtype: {param.dtype}")
        else:
            print(f"  {i:3d}. {key:60s} | type: {type(param)}")
    
    # 分析不同模块
    print("\n" + "-" * 80)
    print("模块统计")
    print("-" * 80)
    
    module_counts = {}
    for key in param_keys:
        module_name = key.split('.')[0]
        module_counts[module_name] = module_counts.get(module_name, 0) + 1
    
    print(f"\n{'模块名':<30s} | 参数数量")
    print("-" * 50)
    for module, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        print(f"{module:<30s} | {count:6d}")
    
    # 特定检查
    print("\n" + "=" * 80)
    print("特定模块检查")
    print("=" * 80)
    
    # 检查 backbone
    backbone_keys = [k for k in param_keys if 'blocks' in k or 'patch_embed' in k or 'pos_embed' in k]
    print(f"\n✓ Backbone 参数: {len(backbone_keys)} 个")
    if backbone_keys:
        print("  示例:")
        for key in backbone_keys[:5]:
            print(f"    - {key}")
    
    # 检查检测头
    detection_keys = [k for k in param_keys if any(x in k for x in ['detector', 'rpn', 'roi', 'box', 'class'])]
    print(f"\n{'✓' if detection_keys else '✗'} 检测头参数: {len(detection_keys)} 个")
    if detection_keys:
        print("  示例:")
        for key in detection_keys[:5]:
            print(f"    - {key}")
    
    # 检查分割头
    segmentation_keys = [k for k in param_keys if any(x in k for x in ['decoder', 'head', 'mask_embed', 'query'])]
    print(f"\n{'✓' if segmentation_keys else '✗'} 分割头参数: {len(segmentation_keys)} 个")
    if segmentation_keys:
        print("  示例:")
        for key in segmentation_keys[:10]:
            print(f"    - {key}")
    
    # 检查分类器/输出层
    output_keys = [k for k in param_keys if 'fc' in k or 'classifier' in k or 'linear' in k]
    if output_keys:
        print(f"\n输出层参数:")
        for key in output_keys[:10]:
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                print(f"  - {key}: {tuple(param.shape)}")
    
    # 推断类别数（对于分割模型）
    print("\n" + "=" * 80)
    print("推断配置")
    print("=" * 80)
    
    # 查找输出维度
    potential_output_keys = [
        k for k in param_keys 
        if k.endswith('.weight') and any(x in k for x in ['head', 'decoder', 'classifier', 'fc'])
    ]
    
    if potential_output_keys:
        print("\n可能的输出层:")
        for key in potential_output_keys[:10]:
            param = state_dict[key]
            if isinstance(param, torch.Tensor) and param.dim() >= 2:
                out_dim = param.shape[0]
                print(f"  - {key}")
                print(f"    输出维度: {out_dim}")
                
                # 推测类别数
                if checkpoint_type == "segmentation":
                    if out_dim == 150:
                        print(f"    ✓ ADE20K (150 类)")
                    elif out_dim == 21:
                        print(f"    ✓ PASCAL VOC (21 类)")
                    elif out_dim == 19:
                        print(f"    ✓ Cityscapes (19 类)")
                    else:
                        print(f"    ? 未知数据集 ({out_dim} 类)")
                elif checkpoint_type == "detection":
                    if out_dim == 80 or out_dim == 81:
                        print(f"    ✓ COCO (80 类)")
                    elif out_dim == 20 or out_dim == 21:
                        print(f"    ✓ PASCAL VOC (20 类)")
    
    # 元数据
    print("\n" + "=" * 80)
    print("元数据")
    print("=" * 80)
    
    metadata_keys = [k for k in ckpt.keys() if k != 'model' and k != 'state_dict']
    if metadata_keys:
        print("\n其他键:")
        for key in metadata_keys:
            value = ckpt[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"  - {key}: {value}")
            elif isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} items")
                if len(value) <= 10:
                    for k, v in value.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  - {key}: {type(value)}")
    
    print("\n" + "=" * 80)
    print("✓ 验证完成")
    print("=" * 80)
    
    return True


def main():
    checkpoints = {
        "backbone": "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth",
        "detection": "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth",
        "segmentation": "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth",
    }
    
    for ckpt_type, ckpt_path in checkpoints.items():
        verify_checkpoint(ckpt_path, ckpt_type)
        print("\n\n")


if __name__ == "__main__":
    main()