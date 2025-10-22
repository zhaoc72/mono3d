# mono3d

Zero-shot instance segmentation toolkit that wires together [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
for proposal generation and [facebookresearch/sam2](https://github.com/facebookresearch/sam2) for promptable segmentation.
The resulting masks are saved in a format that can be consumed by downstream 3D reconstruction pipelines such as
3D Gaussian Splatting.

## Features

- TorchHub-compatible DINOv3 loader with automatic attention heatmap extraction.
- Prompt generation utilities (bounding boxes + positive/negative points) derived from DINOv3 attention.
- SAM2 wrapper built on top of the official Hugging Face integration for batched prompt decoding.
- Modular post-processing and reconstruction export utilities for easy integration with 3DGS workflows.
- CLI pipeline supporting single images, directories, and videos.

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

Set the Virtual KITTI 2 dataset root to `/media/pc/D/datasets/vkitti2` for the helper scripts below.

## Running Inference

```bash
python -m src.inference_pipeline \
  image \
  --input path/to/image.png \
  --output outputs/run_001 \
  --config configs/model_config.yaml \
  --prompt-config configs/prompt_config.yaml
```

For directory processing replace `image` with `directory` and provide a directory path, or use `video` for video files.

To run on the VKITTI2 dataset end-to-end:

```bash
./scripts/run_vkitti.sh \
  /media/pc/D/datasets/vkitti2 \
  outputs/vkitti_scene_masks \
  configs/model_config.yaml \
  configs/prompt_config.yaml \
  Camera_0
```

Advanced usage allows filtering scenes/clones and limiting frame counts:

```bash
python -m src.inference_pipeline vkitti \
  --input /media/pc/D/datasets/vkitti2 \
  --output outputs/vkitti_subset \
  --config configs/model_config.yaml \
  --vkitti-scenes Scene01 Scene02 \
  --vkitti-clones clone_0001 clone_0002 \
  --vkitti-limit 200
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

- Set `pipeline.input_size` to resize inputs before inference when speed is critical.
- Adjust `pipeline.max_prompts_per_batch` to balance speed vs. memory usage during SAM2 decoding.
- Modify `reconstruction.mask_extension` to control whether masks are stored as PNG or NPY arrays.
- Ensure the SAM2 configuration YAML matches the checkpoint; the default expects
  `/media/pc/D/zhaochen/MonoSGS-Prior/checkpoints/pretrained/sam2.1_hiera_large.yaml`.
