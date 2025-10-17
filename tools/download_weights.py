#!/usr/bin/env python3
"""Download pretrained model weights

This script downloads all necessary pretrained weights for Mono3D.
"""

import argparse
import logging
from pathlib import Path
import urllib.request
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


WEIGHTS = {
    # DINOv3 (fallback URLs point to DINOv2 weights; rename upon download)
    'dinov3_vits16': {
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'filename': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'size_mb': 270,
    },
    'dinov3_vits16plus': {
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'filename': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'size_mb': 270,
    },
    'dinov3_vitb14': {
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'filename': 'dinov3_vitb14_pretrain.pth',
        'size_mb': 330,
    },
    # SAM2.1 variants
    'sam2_tiny': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'filename': 'sam2.1_hiera_tiny.pt',
        'size_mb': 180,
    },
    'sam2_small': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'filename': 'sam2.1_hiera_small.pt',
        'size_mb': 240,
    },
    'sam2_base_plus': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'filename': 'sam2.1_hiera_base_plus.pt',
        'size_mb': 320,
    },
    'sam2_large': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        'filename': 'sam2.1_hiera_large.pt',
        'size_mb': 870,
    },
    # Depth Anything V2 variants
    'depth_anything_vits': {
        'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
        'filename': 'depth_anything_v2_vits.pth',
        'size_mb': 270,
    },
    'depth_anything_vitb': {
        'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
        'filename': 'depth_anything_v2_vitb.pth',
        'size_mb': 430,
    },
    'depth_anything_vitl': {
        'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        'filename': 'depth_anything_v2_vitl.pth',
        'size_mb': 1320,
    },
}


def download_file(url: str, filepath: Path, size_mb: float):
    """Download a file with progress bar
    
    Args:
        url: Download URL
        filepath: Local file path
        size_mb: Expected size in MB
    """
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        sys.stdout.write(f'\rDownloading: {percent:.1f}% ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)')
        sys.stdout.flush()
    
    log.info(f"Downloading {filepath.name} (~{size_mb} MB)")
    
    try:
        urllib.request.urlretrieve(url, filepath, reporthook)
        sys.stdout.write('\n')
        log.info(f"Successfully downloaded to {filepath}")
    except Exception as e:
        log.error(f"Failed to download: {e}")
        if filepath.exists():
            filepath.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/pretrained',
        help='Output directory for weights'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(WEIGHTS.keys()) + ['all'],
        default=['all'],
        help='Models to download'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to download
    if 'all' in args.models:
        models_to_download = WEIGHTS.keys()
    else:
        models_to_download = args.models
    
    # Download
    for model_name in models_to_download:
        weight_info = WEIGHTS[model_name]
        filepath = output_dir / weight_info['filename']
        
        # Check if already exists
        if filepath.exists() and not args.force:
            log.info(f"Skipping {model_name} (already exists)")
            continue
        
        # Download
        try:
            download_file(
                weight_info['url'],
                filepath,
                weight_info['size_mb']
            )
        except Exception as e:
            log.error(f"Failed to download {model_name}: {e}")
            continue
    
    log.info("Download complete!")
    log.info(f"Weights saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()