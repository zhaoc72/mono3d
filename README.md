# mono3d

Zero-shot instance segmentation toolkit that wires together
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) for semantic understanding and
[facebookresearch/sam2](https://github.com/facebookresearch/sam2) for promptable segmentation. The refactored pipeline
follows the technical plan outlined in the user brief: DINOv3 features drive objectness, detection and segmentation
adapters, which are fused into prompts for SAM2 to refine into class-aware foreground masks.

## Features

- Official DINOv3 integration with multi-layer fusion, PCA, and objectness scoring.
- Lightweight detection/segmentation adapters that can load official checkpoints or operate in random-init mode for
  experimentation.
- Fusion utilities that intersect detection boxes, segmentation probabilities, and objectness to isolate foreground
  regions before SAM2 refinement.
- Simple prompt builder (points/boxes/mask seeds) and SAM2 wrapper that works with the official Meta release.
- Streamlined CLI pipeline supporting single images or directories with optional intermediate diagnostics.

## Getting Started

```bash
# clone dependencies (inside your workspace)
git clone https://github.com/facebookresearch/dinov3.git \
  /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/facebookresearch/dinov3
pip install -r /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/facebookresearch/dinov3/requirements.txt

git clone https://github.com/facebookresearch/sam2.git \
  /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/facebookresearch/sam2
pip install -e /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/facebookresearch/sam2

pip install -r requirements.txt  # optional: add project-specific deps

# download weights (paths match configs/model_config.yaml defaults)
wget -O /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/dinov3_vitl16_lvd1689m.pth \
  https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_lvd1689m.pth
wget -O /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
cp /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/facebookresearch/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  /media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.yaml
```

Update `configs/model_config.yaml` if your environment uses different paths. The defaults assume:

- `/media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/dinov3_vitl16_lvd1689m.pth`
- `/media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.pt`
- `/media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.yaml`

Set the Virtual KITTI 2 dataset root to `/media/pc/D/datasets/vkitti2` if you want to reproduce the examples below.

## Running Inference

```bash
python -m src image \
  --input path/to/image.png \
  --output outputs/run_001 \
  --config configs/model_config.yaml
```

For directory processing replace `image` with `directory` and provide a directory path.

### Zero-shot foreground instance pipeline

The refactored codebase implements the design brief directly:

1. **DINOv3 backbone** (official repo) emits multi-layer patch tokens, attention, and an objectness map.
2. **Detection adapter** provides coarse bounding boxes and class probabilities.
3. **Segmentation adapter** outputs per-class probability maps over the processed resolution.
4. The fusion module combines detection, segmentation, and objectness to isolate class-aware foreground regions and
   splits them with connected components.
5. The prompt builder creates points/boxes/mask seeds which SAM2 refines into sharp instance masks.

All steps operate in zero-shot mode—only the published checkpoints are required. To export intermediate fusion
artifacts during inference:

```bash
python -m src image \
  --input /media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg \
  --output outputs/adapter \
  --config configs/model_config.yaml \
  --visualize-intermediate
```

## Project Layout

```
configs/                  # YAML configs for models and prompting
checkpoints/              # Downloaded weights (gitignored)
src/                      # Python source modules
scripts/                  # Shell helpers for batch jobs
outputs/                  # Default output root for masks & recon inputs
```

## Notes

- Modify `configs/model_config.yaml` to adjust fusion thresholds, prompt options, and post-processing.
- Ensure the SAM2 configuration YAML matches the checkpoint; the default expects
  `/media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.yaml`.
