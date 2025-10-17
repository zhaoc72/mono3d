"""特征缓存系统

提供LMDB和PyTorch两种缓存后端，用于加速数据加载。
"""

import lmdb
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import logging
import hashlib

log = logging.getLogger(__name__)


def build_cache(cfg):
    """构建特征缓存
    
    Args:
        cfg: 配置对象，需包含:
            - data.name: 数据集名称
            - data.cache.cache_dir: 缓存目录
            - data.cache.features: 要缓存的特征列表
            - data.cache.backend: 缓存后端 (lmdb/pt)
    """
    log.info(f"Building cache for dataset: {cfg.data.name}")
    
    cache_dir = Path(cfg.data.cache.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    backend = cfg.data.cache.get('backend', 'lmdb')
    
    if backend == 'lmdb':
        _build_lmdb_cache(cfg, cache_dir)
    elif backend == 'pt':
        _build_pt_cache(cfg, cache_dir)
    else:
        raise ValueError(f"Unknown cache backend: {backend}")


def _build_lmdb_cache(cfg, cache_dir: Path):
    """构建LMDB缓存"""
    # 初始化LMDB环境
    db_path = cache_dir / "features.lmdb"
    
    # 计算合理的map_size
    map_size = 1024**4  # 1TB (LMDB会自动处理)
    
    env = lmdb.open(
        str(db_path),
        map_size=map_size,
        max_dbs=3,  # 3个子数据库：train/val/test
        writemap=True,
        map_async=True,
        meminit=False,
    )
    
    # 创建子数据库
    db_map = {}
    for split in ['train', 'val', 'test']:
        db_map[split] = env.open_db(split.encode('utf-8'))
    
    # 加载特征提取模型
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    models = _load_feature_models(cfg, device)
    
    # 遍历数据集提取特征
    from .datasets import build_dataset
    
    for split in ['train', 'val', 'test']:
        log.info(f"Processing {split} split...")
        
        try:
            dataset = build_dataset(cfg, split=split, transform=None)
        except Exception as e:
            log.warning(f"Cannot load {split} split: {e}")
            continue
        
        if len(dataset) == 0:
            log.warning(f"Empty {split} split, skipping")
            continue
        
        db = db_map[split]
        
        # 批处理
        batch_size = cfg.get('cache_batch_size', 8)
        
        with env.begin(write=True, db=db) as txn:
            for idx in tqdm(range(len(dataset)), desc=f"Caching {split}"):
                try:
                    sample = dataset[idx]
                    
                    # 提取特征
                    features = _extract_features(sample, models, device, cfg)
                    
                    # 添加元数据
                    features['meta'] = {
                        'image_id': sample.get('image_id', idx),
                        'category': sample.get('category', 'unknown'),
                        'split': split,
                    }
                    
                    # 序列化并存储
                    key = f"{idx:08d}".encode('utf-8')
                    value = pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)
                    txn.put(key, value)
                
                except Exception as e:
                    log.error(f"Error processing sample {idx}: {e}")
                    continue
        
        log.info(f"Cached {len(dataset)} samples for {split}")
    
    env.close()
    log.info(f"Cache built successfully at {db_path}")


def _build_pt_cache(cfg, cache_dir: Path):
    """构建PyTorch .pt文件缓存"""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    models = _load_feature_models(cfg, device)
    
    from .datasets import build_dataset
    
    for split in ['train', 'val', 'test']:
        log.info(f"Processing {split} split...")
        
        try:
            dataset = build_dataset(cfg, split=split, transform=None)
        except Exception as e:
            log.warning(f"Cannot load {split} split: {e}")
            continue
        
        if len(dataset) == 0:
            continue
        
        # 创建分割目录
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        cache = PTCache(cache_dir, split)
        
        for idx in tqdm(range(len(dataset)), desc=f"Caching {split}"):
            try:
                sample = dataset[idx]
                features = _extract_features(sample, models, device, cfg)
                
                features['meta'] = {
                    'image_id': sample.get('image_id', idx),
                    'category': sample.get('category', 'unknown'),
                    'split': split,
                }
                
                cache.save(idx, features)
            
            except Exception as e:
                log.error(f"Error processing sample {idx}: {e}")
                continue
        
        cache.finalize()
        log.info(f"Cached {len(dataset)} samples for {split}")


def _load_feature_models(cfg, device):
    """加载特征提取模型"""
    from ..registry import build
    
    models = {}
    
    features_to_cache = cfg.data.cache.features
    
    if 'dino' in features_to_cache:
        log.info("Loading DINOv3...")
        models['dino'] = build("model", "dinov3", **cfg.model.dinov3).to(device).eval()
    
    if 'depth' in features_to_cache:
        log.info("Loading Depth Anything...")
        models['depth'] = build("model", "depth_anything", **cfg.model.depth).to(device).eval()
    
    if 'mask' in features_to_cache:
        log.info("Loading SAM2...")
        models['sam'] = build("model", "sam2", **cfg.model.sam2).to(device).eval()
    
    return models


def _extract_features(sample, models, device, cfg):
    """提取特征"""
    features = {}
    
    # 准备图像
    image = sample['image']
    
    # 转换为tensor
    if not isinstance(image, torch.Tensor):
        from torchvision.transforms import ToTensor
        image = ToTensor()(image)
    
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # DINOv3特征
        if 'dino' in models:
            try:
                dino_features = models['dino'](image)
                features['dino'] = dino_features.cpu()
            except Exception as e:
                log.warning(f"Failed to extract DINO features: {e}")
        
        # 深度
        if 'depth' in models:
            try:
                depth_pred = models['depth'](image)
                features['depth'] = depth_pred.cpu()
            except Exception as e:
                log.warning(f"Failed to extract depth: {e}")
        
        # 掩码
        if 'mask' in models:
            try:
                # 这里需要提供提示（如边界框或点）
                # 简化：如果样本中已有mask则跳过
                if 'mask' not in sample:
                    mask_pred = models['sam'](image)
                    features['mask'] = mask_pred.cpu()
            except Exception as e:
                log.warning(f"Failed to extract mask: {e}")
    
    # 保存原始数据（可选）
    if cfg.data.cache.get('save_original', False):
        features['image'] = image.cpu()
        if 'mask' in sample:
            features['mask'] = sample['mask']
    
    return features


class LMDBCache:
    """LMDB缓存读取器"""
    
    def __init__(self, db_path: Path, split: str = 'train', readonly: bool = True):
        self.db_path = Path(db_path)
        self.split = split
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Cache database not found: {db_path}")
        
        # 打开环境
        self.env = lmdb.open(
            str(db_path),
            readonly=readonly,
            lock=readonly,
            readahead=True,
            meminit=False,
            max_dbs=3,
        )
        
        # 打开子数据库
        self.db = self.env.open_db(split.encode('utf-8'))
        
        # 获取长度
        with self.env.begin(db=self.db) as txn:
            stats = txn.stat()
            self.length = stats['entries']
        
        log.info(f"Loaded LMDB cache with {self.length} entries ({split})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """读取缓存的特征"""
        key = f"{idx:08d}".encode('utf-8')
        
        with self.env.begin(db=self.db, write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {idx} not found in cache")
            
            features = pickle.loads(value)
        
        return features
    
    def close(self):
        """关闭数据库"""
        if self.env:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PTCache:
    """PyTorch .pt文件缓存（轻量替代方案）"""
    
    def __init__(self, cache_dir: Path, split: str = 'train'):
        self.cache_dir = Path(cache_dir) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        
        # 索引文件
        self.index_file = self.cache_dir / "index.txt"
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.files = [line.strip() for line in f]
        else:
            self.files = []
        
        log.info(f"Loaded PT cache with {len(self.files)} entries ({split})")
    
    def save(self, idx: int, features: Dict[str, Any]):
        """保存特征"""
        filename = f"{idx:08d}.pt"
        filepath = self.cache_dir / filename
        
        torch.save(features, filepath)
        
        if filename not in self.files:
            self.files.append(filename)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """读取特征"""
        filename = f"{idx:08d}.pt"
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise KeyError(f"Cache file {filepath} not found")
        
        return torch.load(filepath, map_location='cpu')
    
    def __len__(self):
        return len(self.files)
    
    def finalize(self):
        """完成缓存构建，写入索引"""
        with open(self.index_file, 'w') as f:
            for filename in sorted(self.files):
                f.write(f"{filename}\n")
        
        log.info(f"Finalized cache with {len(self.files)} files")


class CacheManager:
    """缓存管理器
    
    提供缓存的创建、加载、清理等管理功能。
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        info = {
            'cache_dir': str(self.cache_dir),
            'exists': False,
            'backend': None,
            'size_bytes': 0,
            'splits': {},
        }
        
        # 检查LMDB
        lmdb_path = self.cache_dir / "features.lmdb"
        if lmdb_path.exists():
            info['exists'] = True
            info['backend'] = 'lmdb'
            info['size_bytes'] = self._get_dir_size(lmdb_path)
            
            # 读取每个分割的大小
            try:
                env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
                for split in ['train', 'val', 'test']:
                    try:
                        db = env.open_db(split.encode('utf-8'))
                        with env.begin(db=db) as txn:
                            stats = txn.stat()
                            info['splits'][split] = stats['entries']
                    except:
                        pass
                env.close()
            except:
                pass
        
        # 检查PT
        elif any(self.cache_dir.glob('*/index.txt')):
            info['exists'] = True
            info['backend'] = 'pt'
            info['size_bytes'] = self._get_dir_size(self.cache_dir)
            
            for split_dir in self.cache_dir.iterdir():
                if split_dir.is_dir():
                    index_file = split_dir / 'index.txt'
                    if index_file.exists():
                        with open(index_file) as f:
                            info['splits'][split_dir.name] = sum(1 for _ in f)
        
        return info
    
    def _get_dir_size(self, path: Path) -> int:
        """计算目录大小"""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
    
    def clear_cache(self):
        """清空缓存"""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Cleared cache at {self.cache_dir}")
    
    def verify_cache(self, dataset_size: Dict[str, int]) -> bool:
        """验证缓存完整性
        
        Args:
            dataset_size: 每个分割的预期大小 {'train': N, 'val': M, ...}
            
        Returns:
            是否完整
        """
        info = self.get_cache_info()
        
        if not info['exists']:
            return False
        
        for split, size in dataset_size.items():
            if split not in info['splits']:
                log.warning(f"Split {split} not found in cache")
                return False
            
            cached_size = info['splits'][split]
            if cached_size != size:
                log.warning(
                    f"Split {split} size mismatch: "
                    f"expected {size}, got {cached_size}"
                )
                return False
        
        log.info("Cache verification passed")
        return True


def load_cache(cache_dir: Path, split: str = 'train', backend: str = 'auto'):
    """加载缓存
    
    Args:
        cache_dir: 缓存目录
        split: 数据分割
        backend: 后端类型 ('auto', 'lmdb', 'pt')
        
    Returns:
        缓存对象
    """
    cache_dir = Path(cache_dir)
    
    if backend == 'auto':
        # 自动检测
        if (cache_dir / "features.lmdb").exists():
            backend = 'lmdb'
        elif (cache_dir / split / "index.txt").exists():
            backend = 'pt'
        else:
            raise FileNotFoundError(f"No cache found at {cache_dir}")
    
    if backend == 'lmdb':
        return LMDBCache(cache_dir / "features.lmdb", split=split)
    elif backend == 'pt':
        return PTCache(cache_dir, split=split)
    else:
        raise ValueError(f"Unknown backend: {backend}")