# Mono3D 多模态流水线框架

本仓库提供一个统一的推理与重建编排框架，目标是在单目 3D 感知场景中整合多个视觉模块：

- **DINOv3**：ViT-7B 主干，支持检测与语义分割 Adapter，并结合 Accelerate 自动在四张 4090 间切分；所有推理统一使用 float16；
- **SAM2**（规划中）：生成掩码与提示；
- **3D Gaussian Splatting**（规划中）：完成重建与可视化。

新的目录结构将已有的 DINOv3 推理逻辑纳入更通用的项目骨架，后续扩展其它模块时无需改动核心调度代码。

## 目录结构

```
configs/
  model_config.yaml                # Mono3D 顶层 YAML 配置
mono3d/
  core/
    configuration.py               # 数据类定义 & YAML 解析
    runner.py                      # 多流水线调度器
  pipelines/
    dinov3/                        # DINOv3 类型标识
    sam2/                          # SAM2 类型标识（占位）
    reconstruction/                # 3DGS 类型标识（占位）
  __init__.py                      # 对外导出核心 API
  ...
tools/
  run_project.py                   # 通用命令行入口
  run_dinov3_pipeline.py           # 兼容入口，转发到 run_project.py
```

## 环境准备

1. 保持本仓库与官方 DINOv3 代码处于同一级目录，例如：
   ```
   /media/pc/D/zhaochen/mono3d/
   ├── dinov3/
   └── mono3d/
   ```
2. 安装必要依赖：
   ```bash
   pip install accelerate transformers pyyaml
   ```
3. 若计划使用 SAM2 或 3D Gaussian Splatting，请将对应仓库克隆至同级目录，后续只需在 YAML 中补全路径。
4. 建议结合 Hugging Face Accelerate，启用 `device_map="auto"` 在四张 RTX 4090 上运行 ViT-7B。

## 准备 YAML 配置

顶层配置位于 `configs/model_config.yaml`，统一描述多个流水线。关键区块说明：

- `project`：项目名称与描述，便于区分实验；
- `context`：全局模板变量（如输入/输出目录），可被所有流水线引用；
- `execution`：Python 与 Accelerate 的默认执行方式；
- `environment`：对所有任务生效的环境变量；
- `pipelines`：流水线列表，目前包含 DINOv3、SAM2（占位）、3D 重建（占位）。其中 DINOv3 默认调用 `tools/dinov3_accelerate_inference.py`，通过 `device_map="auto"` 自动分配模型参数。

DINOv3 配置示例片段：

```yaml
pipelines:
  - name: dinov3
    type: dinov3
    repo_path: ../dinov3
    environment:
      PYTHONPATH: "{repo_path}:{PYTHONPATH}"
    context:
      backbone_checkpoint: /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth
      detection_checkpoint: /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth
      segmentation_checkpoint: /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth
      input_path: "{input_path}"
      output_dir: "{output_dir}"
    tasks:
      - name: detection
        script: ../mono3d/tools/dinov3_accelerate_inference.py
        args:
          - --repo
          - "{repo_path}"
          - --task
          - detection
          - --backbone
          - "{backbone_checkpoint}"
          - --detection-adapter
          - "{detection_checkpoint}"
          - --input
          - "{input_path}"
          - --output
          - "{output_dir}/detection"
          - --device-map
          - auto
          - --save-visuals
      - name: segmentation
        script: ../mono3d/tools/dinov3_accelerate_inference.py
        args:
          - --repo
          - "{repo_path}"
          - --task
          - segmentation
          - --backbone
          - "{backbone_checkpoint}"
          - --segmentation-adapter
          - "{segmentation_checkpoint}"
          - --input
          - "{input_path}"
          - --output
          - "{output_dir}/segmentation"
          - --device-map
          - auto
          - --save-visuals
          - --color-map
          - Spectral
      - name: detection+segmentation
        script: ../mono3d/tools/dinov3_accelerate_inference.py
        args:
          - --repo
          - "{repo_path}"
          - --task
          - both
          - --backbone
          - "{backbone_checkpoint}"
          - --detection-adapter
          - "{detection_checkpoint}"
          - --segmentation-adapter
          - "{segmentation_checkpoint}"
          - --input
          - "{input_path}"
          - --output
          - "{output_dir}/joint"
          - --device-map
          - auto
          - --save-visuals
          - --color-map
          - Spectral
```

如需调整显存占用，可通过 `--max-memory` 指定 GiB 限制，例如 `--max-memory 22`。脚本会调用 Accelerate 的 `infer_auto_device_map`，在四张 RTX 4090 间自动切分 ViT-7B 权重。

每条流水线包含若干 `tasks`，对应需要执行的官方脚本；任务参数仍支持格式化模板，方便根据上下文拼接路径。

### 覆盖配置

运行时可通过 `--set` 修改上下文变量：

- `--set input_path=/path/to/image.jpg` 会覆盖全局 `context.input_path`；
- `--set pipeline:dinov3.backbone_checkpoint=/path/to.pth` 仅影响 DINOv3 流水线的对应参数。

## 启动流程

```bash
python tools/run_project.py --config configs/model_config.yaml
```

常用参数：

- `--dry-run`：仅打印命令，不真正执行；
- `--verbose`：输出调试日志；
- `--list`：查看配置中所有流水线及启用状态；
- `--only dinov3 sam2`：只运行指定流水线。

历史脚本 `tools/run_dinov3_pipeline.py` 仍可使用，会自动转发到新的通用入口。

## 单张图像快速验证

要验证官方预训练的 ViT-7B backbone 与检测/分割 Adapter 是否能够在四张 RTX 4090 上正确推理，可直接调用 Accelerate 脚本处理 COCO2017 中的测试图像 `/media/pc/D/datasets/coco2017/train2017/000000000532.jpg`：

```bash
# 1. 仅执行 DINOv3 检测（附带可视化）
python tools/dinov3_accelerate_inference.py \
  --repo /media/pc/D/zhaochen/mono3d/dinov3 \
  --task detection \
  --backbone /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth \
  --detection-adapter /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth \
  --input /media/pc/D/datasets/coco2017/train2017/000000000532.jpg \
  --output /media/pc/D/zhaochen/mono3d/outputs/detection \
  --device-map auto \
  --save-visuals

# 2. 仅执行 DINOv3 语义分割（保存伪彩叠加图）
python tools/dinov3_accelerate_inference.py \
  --repo /media/pc/D/zhaochen/mono3d/dinov3 \
  --task segmentation \
  --backbone /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth \
  --segmentation-adapter /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth \
  --input /media/pc/D/datasets/coco2017/train2017/000000000532.jpg \
  --output /media/pc/D/zhaochen/mono3d/outputs/segmentation \
  --device-map auto \
  --save-visuals \
  --color-map Spectral

# 3. 同时运行检测 + 分割，共享 backbone 并保存全部可视化
python tools/dinov3_accelerate_inference.py \
  --repo /media/pc/D/zhaochen/mono3d/dinov3 \
  --task both \
  --backbone /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth \
  --detection-adapter /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth \
  --segmentation-adapter /media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth \
  --input /media/pc/D/datasets/coco2017/train2017/000000000532.jpg \
  --output /media/pc/D/zhaochen/mono3d/outputs/joint \
  --device-map auto \
  --save-visuals \
  --color-map Spectral
```

推理结束后：

- 检测结果会以 `*_detections.json` 形式保存在 `outputs/detection/`，若启用 `--save-visuals` 还会输出 `*_detection_overlay.png`；
- 分割结果保存在 `outputs/segmentation/`，包含 16 位索引图 (`*_segmentation.png`) 与伪彩叠加图 (`*_overlay.png`)；
- 当使用 `--task both` 时，检测与分割分别输出到 `outputs/joint/detection/` 与 `outputs/joint/segmentation/`，共用同一份 backbone 权重。

## torch.hub 示例

若需在自定义脚本中直接加载官方模型，可参考以下示例：

```python
import torch
from torchvision.transforms import v2
from PIL import Image
from functools import partial
from dinov3.eval.segmentation.inference import make_inference

REPO_DIR = "/media/pc/D/zhaochen/mono3d/dinov3"
BACKBONE = f"{REPO_DIR}/checkpoints/dinov3_vit7b16_lvd1689m.pth"
DET_ADAPTER = f"{REPO_DIR}/checkpoints/dinov3_vit7b16_coco.pth"
SEG_ADAPTER = f"{REPO_DIR}/checkpoints/dinov3_vit7b16_ade20k.pth"

# 检测模型
detector = torch.hub.load(
    REPO_DIR,
    "dinov3_vit7b16_de",
    source="local",
    weights=DET_ADAPTER,
    backbone_weights=BACKBONE,
)

# 分割模型
segmentor = torch.hub.load(
    REPO_DIR,
    "dinov3_vit7b16_ms",
    source="local",
    weights=SEG_ADAPTER,
    backbone_weights=BACKBONE,
)
```

通过 `make_inference` 可获得最终语义分割结果。若想单独测试多卡推理，可参考前文的三种命令示例；脚本会基于 Accelerate 自动推断 `device_map`，在四张 RTX 4090 之间切分 ViT-7B 模型。

## 后续规划

- 接入 **SAM2** 推理脚本，复用当前调度器实现图像掩码生成；
- 整合 **3D Gaussian Splatting** 重建流程，实现单目输入到 3D 场景的端到端流水线；
- 根据需求扩展更多任务类型，例如特征提取、点云配准等。

## 许可证

本仓库默认使用 MIT 许可证，外部依赖（如 DINOv3、SAM2、3DGS）遵循其原始许可协议。
