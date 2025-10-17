#!/usr/bin/env python3
"""Export models to various formats

Supports:
- ONNX
- TorchScript
- TensorRT (if available)
"""

import argparse
import logging
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mono3d.models import GaussianModel
from mono3d.utils.io import load_gaussian_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 256, 256)
):
    """Export model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input tensor shape
    """
    log.info(f"Exporting to ONNX: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13,
    )
    
    log.info(f"ONNX model saved to {output_path}")


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 256, 256)
):
    """Export model to TorchScript
    
    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input tensor shape
    """
    log.info(f"Exporting to TorchScript: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Trace model
    traced = torch.jit.trace(model, dummy_input)
    
    # Save
    traced.save(str(output_path))
    
    log.info(f"TorchScript model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export models")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exports',
        help='Output directory'
    )
    parser.add_argument(
        '--format',
        choices=['onnx', 'torchscript', 'all'],
        default='all',
        help='Export format'
    )
    parser.add_argument(
        '--input-shape',
        nargs=4,
        type=int,
        default=[1, 3, 256, 256],
        help='Input shape (B C H W)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    log.info(f"Loading model from {args.model_path}")
    model = load_gaussian_model(args.model_path)
    model.eval()
    
    # Export
    model_name = Path(args.model_path).stem
    
    if args.format in ['onnx', 'all']:
        output_path = output_dir / f"{model_name}.onnx"
        export_to_onnx(model, output_path, tuple(args.input_shape))
    
    if args.format in ['torchscript', 'all']:
        output_path = output_dir / f"{model_name}.pt"
        export_to_torchscript(model, output_path, tuple(args.input_shape))
    
    log.info("Export complete!")


if __name__ == '__main__':
    main()