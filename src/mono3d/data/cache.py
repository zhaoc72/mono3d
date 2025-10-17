"""特征缓存系统"""
import lmdb
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import logging

from ..registry import build

log = logging.getLogger(__name__)

def build_cache(cfg):
    """构建特征缓存"""
    log.info(f"Building cache for dataset: {cfg.data.name}")
    
    cache_dir = Path(cfg.data.cache.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化LMDB环境
    db_path = cache_dir / "features.lmdb"
    env = lmdb.open(
        str(db_path),
        map_size=1024**4,  # 1TB
        max_dbs=3,         # 3个子数据库：train/val/test
        writemap=True,
        map_async=True,
    )
    
    # 创建子数据库
    db_train = env.open_db(b'train')
    db_val = env.open_db(b'val')
    db_test = env.open_db(b'test')
    
    db_map = {
        'train': db_train,
        'val': db_val,
        'test': db_test,
    }
    
    # 加载特征提取模型
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    models = {}
    if 'dino' in cfg.data.cache.features:
        models['dino'] = build("model", "dinov3", **cfg.model.dinov3).to(device).eval()
    if 'depth' in cfg.data.cache.features:
        models['depth'] = build("model", "depth_anything", **cfg.model.depth).to(device).eval()
    if 'mask' in cfg.data.cache.features:
        models['sam'] = build("model", "sam2", **cfg.model.sam2).to(device).eval()
    
    # 遍历数据集提取特征
    from .datasets import build_dataset
    
    for split in ['train', 'val', 'test']:
        log.info(f"Processing {split} split...")
        
        dataset = build_dataset(cfg, split=split)
        db = db_map[split]
        
        with env.begin(write=True, db=db) as txn:
            for idx in tqdm(range(len(dataset)), desc=f"Caching {split}"):
                sample = dataset[idx]
                
                # 提取特征
                features = {}
                image = sample['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if 'dino' in models:
                        features['dino'] = models['dino'](image).cpu()
                    if 'depth' in models:
                        features['depth'] = models['depth'](image).cpu()
                    if 'mask' in models:
                        features['mask'] = models['sam'](image).cpu()
                
                # 添加元数据
                features['meta'] = {
                    'image_id': sample.get('image_id', idx),
                    'category': sample.get('category', 'unknown'),
                }
                
                # 序列化并存储
                key = f"{idx:08d}".encode('utf-8')
                value = pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, value)
        
        log.info(f"Cached {len(dataset)} samples for {split}")
    
    env.close()
    log.info(f"Cache built successfully at {db_path}")


class LMDBCache:
    """LMDB缓存读取器"""
    
    def __init__(self, db_path: Path, split: str = 'train', readonly: bool = True):
        self.db_path = db_path
        self.split = split
        
        # 打开环境
        self.env = lmdb.open(
            str(db_path),
            readonly=readonly,
            lock=readonly,
            readahead=True,
            meminit=False,
        )
        
        # 打开子数据库
        self.db = self.env.open_db(split.encode('utf-8'))
        
        # 获取长度
        with self.env.begin(db=self.db) as txn:
            self.length = txn.stat()['entries']
    
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
    
    def __del__(self):
        self.close()


class PTCache:
    """PyTorch .pt文件缓存（轻量替代方案）"""
    
    def __init__(self, cache_dir: Path, split: str = 'train'):
        self.cache_dir = cache_dir / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        
        # 索引文件
        self.index_file = self.cache_dir / "index.txt"
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.files = [line.strip() for line in f]
        else:
            self.files = []
    
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
            for filename in self.files:
                f.write(f"{filename}\n")