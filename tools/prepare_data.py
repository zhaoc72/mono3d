#!/usr/bin/env python3
"""Data preparation script

Prepares datasets for training by:
- Downloading datasets
- Extracting archives
- Organizing file structure
- Computing statistics
"""

import argparse
import logging
from pathlib import Path
import subprocess
import shutil
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


DATASETS = {
    'pix3d': {
        'url': 'http://pix3d.csail.mit.edu/data/pix3d.zip',
        'filename': 'pix3d.zip',
        'extract_dir': 'Pix3D',
        'size_gb': 2.5,
    },
    'co3dv2': {
        'url': None,  # Requires registration
        'instructions': 'Visit https://github.com/facebookresearch/co3d and follow download instructions',
        'extract_dir': 'CO3Dv2',
        'size_gb': 150,
    },
}


def download_dataset(dataset_name: str, data_dir: Path):
    """Download dataset
    
    Args:
        dataset_name: Name of dataset
        data_dir: Data directory
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_info = DATASETS[dataset_name]
    
    # Check if requires manual download
    if dataset_info['url'] is None:
        log.warning(f"Dataset {dataset_name} requires manual download")
        log.info(dataset_info['instructions'])
        return
    
    # Create directory
    extract_dir = data_dir / dataset_info['extract_dir']
    if extract_dir.exists():
        log.info(f"Dataset {dataset_name} already exists at {extract_dir}")
        return
    
    # Download
    archive_path = data_dir / dataset_info['filename']
    
    if not archive_path.exists():
        log.info(f"Downloading {dataset_name} (~{dataset_info['size_gb']} GB)")
        log.info(f"URL: {dataset_info['url']}")
        
        subprocess.run([
            'wget',
            '-O', str(archive_path),
            dataset_info['url']
        ], check=True)
    
    # Extract
    log.info(f"Extracting {archive_path}")
    
    if archive_path.suffix == '.zip':
        subprocess.run(['unzip', '-q', str(archive_path), '-d', str(data_dir)], check=True)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        subprocess.run(['tar', '-xzf', str(archive_path), '-C', str(data_dir)], check=True)
    
    log.info(f"Dataset extracted to {extract_dir}")
    
    # Cleanup
    if archive_path.exists():
        log.info(f"Removing archive {archive_path}")
        archive_path.unlink()


def compute_statistics(dataset_name: str, data_dir: Path):
    """Compute dataset statistics
    
    Args:
        dataset_name: Name of dataset
        data_dir: Data directory
    """
    log.info(f"Computing statistics for {dataset_name}")
    
    dataset_info = DATASETS[dataset_name]
    dataset_path = data_dir / dataset_info['extract_dir']
    
    if not dataset_path.exists():
        log.warning(f"Dataset not found at {dataset_path}")
        return
    
    # Count files
    stats = {
        'dataset': dataset_name,
        'path': str(dataset_path),
        'total_files': sum(1 for _ in dataset_path.rglob('*') if _.is_file()),
        'images': sum(1 for _ in dataset_path.rglob('*.jpg')) + \
                  sum(1 for _ in dataset_path.rglob('*.png')),
    }
    
    # Save statistics
    stats_file = data_dir / f"{dataset_name}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    log.info(f"Statistics saved to {stats_file}")
    log.info(f"Total files: {stats['total_files']}")
    log.info(f"Images: {stats['images']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()) + ['all'],
        default=['all'],
        help='Datasets to prepare'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download (only compute stats)'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets
    if 'all' in args.datasets:
        datasets = DATASETS.keys()
    else:
        datasets = args.datasets
    
    # Process each dataset
    for dataset_name in datasets:
        log.info(f"\n{'='*60}")
        log.info(f"Processing {dataset_name}")
        log.info('='*60)
        
        if not args.skip_download:
            try:
                download_dataset(dataset_name, data_dir)
            except Exception as e:
                log.error(f"Failed to download {dataset_name}: {e}")
                continue
        
        try:
            compute_statistics(dataset_name, data_dir)
        except Exception as e:
            log.error(f"Failed to compute statistics for {dataset_name}: {e}")
    
    log.info("\nData preparation complete!")


if __name__ == '__main__':
    main()