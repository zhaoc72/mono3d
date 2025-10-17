"""模型注册和依赖注入系统

提供统一的模型注册、构建和管理接口。
"""

from typing import Dict, Any, Callable, Optional, Type
import logging
from functools import wraps

log = logging.getLogger(__name__)


# 全局注册表
_REGISTRIES = {
    'model': {},
    'dataset': {},
    'transform': {},
    'loss': {},
    'optimizer': {},
    'scheduler': {},
}


def register(registry_name: str, name: Optional[str] = None):
    """注册装饰器
    
    用法:
        @register('model', 'dinov3')
        class DINOv3Model:
            ...
        
        或:
        @register('model')  # 自动使用类名
        class DINOv3Model:
            ...
    
    Args:
        registry_name: 注册表名称 ('model', 'dataset', etc.)
        name: 注册名称（可选，默认使用类/函数名）
    """
    def decorator(obj):
        # 使用提供的名称或自动推断
        register_name = name
        if register_name is None:
            if hasattr(obj, '__name__'):
                register_name = obj.__name__
            else:
                raise ValueError("Cannot infer name, please provide explicitly")
        
        # 注册到对应的注册表
        if registry_name not in _REGISTRIES:
            _REGISTRIES[registry_name] = {}
        
        _REGISTRIES[registry_name][register_name] = obj
        
        log.debug(f"Registered {register_name} in {registry_name} registry")
        
        return obj
    
    return decorator


def build(registry_name: str, name: str, *args, **kwargs):
    """构建对象
    
    用法:
        model = build('model', 'dinov3', backbone='vit_base')
    
    Args:
        registry_name: 注册表名称
        name: 对象名称
        *args, **kwargs: 传递给构造函数的参数
        
    Returns:
        构建的对象实例
    """
    if registry_name not in _REGISTRIES:
        raise ValueError(f"Unknown registry: {registry_name}")
    
    registry = _REGISTRIES[registry_name]
    
    if name not in registry:
        available = ', '.join(registry.keys())
        raise ValueError(
            f"'{name}' not found in {registry_name} registry. "
            f"Available: {available}"
        )
    
    obj_class = registry[name]
    
    try:
        # 实例化
        instance = obj_class(*args, **kwargs)
        log.debug(f"Built {name} from {registry_name} registry")
        return instance
    
    except Exception as e:
        log.error(f"Failed to build {name}: {e}")
        raise


def list_registered(registry_name: str) -> list:
    """列出注册表中的所有名称
    
    Args:
        registry_name: 注册表名称
        
    Returns:
        注册名称列表
    """
    if registry_name not in _REGISTRIES:
        return []
    
    return list(_REGISTRIES[registry_name].keys())


def is_registered(registry_name: str, name: str) -> bool:
    """检查是否已注册
    
    Args:
        registry_name: 注册表名称
        name: 对象名称
        
    Returns:
        是否已注册
    """
    if registry_name not in _REGISTRIES:
        return False
    
    return name in _REGISTRIES[registry_name]


def unregister(registry_name: str, name: str):
    """取消注册
    
    Args:
        registry_name: 注册表名称
        name: 对象名称
    """
    if registry_name in _REGISTRIES and name in _REGISTRIES[registry_name]:
        del _REGISTRIES[registry_name][name]
        log.debug(f"Unregistered {name} from {registry_name} registry")


def clear_registry(registry_name: Optional[str] = None):
    """清空注册表
    
    Args:
        registry_name: 注册表名称（None表示清空所有）
    """
    if registry_name is None:
        for reg in _REGISTRIES:
            _REGISTRIES[reg].clear()
        log.info("Cleared all registries")
    elif registry_name in _REGISTRIES:
        _REGISTRIES[registry_name].clear()
        log.info(f"Cleared {registry_name} registry")


class Registry:
    """注册表类（面向对象接口）
    
    用法:
        MODEL_REGISTRY = Registry('model')
        
        @MODEL_REGISTRY.register('dinov3')
        class DINOv3Model:
            ...
        
        model = MODEL_REGISTRY.build('dinov3', ...)
    """
    
    def __init__(self, name: str):
        self.name = name
        if name not in _REGISTRIES:
            _REGISTRIES[name] = {}
    
    def register(self, name: Optional[str] = None):
        """注册装饰器"""
        return register(self.name, name)
    
    def build(self, name: str, *args, **kwargs):
        """构建对象"""
        return build(self.name, name, *args, **kwargs)
    
    def list(self) -> list:
        """列出所有注册对象"""
        return list_registered(self.name)
    
    def is_registered(self, name: str) -> bool:
        """检查是否已注册"""
        return is_registered(self.name, name)
    
    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)
    
    def __repr__(self) -> str:
        items = self.list()
        return f"Registry({self.name}, {len(items)} items: {items})"


# 创建常用注册表
MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')


def lazy_import(module_path: str, class_name: str):
    """懒加载装饰器
    
    延迟导入模块直到实际需要，减少启动时间。
    
    用法:
        @lazy_import('mono3d.models.frontend', 'DINOv3')
        def get_dinov3():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 动态导入
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 自动注册所有模型（在模块导入时执行）
def auto_register_models():
    """自动注册所有模型"""
    # 这个函数会在 __init__.py 中调用
    # 导入所有模型模块以触发 @register 装饰器
    try:
        from . import models
        # models模块的__init__.py会导入所有子模块
    except ImportError as e:
        log.warning(f"Failed to auto-register models: {e}")


def auto_register_datasets():
    """自动注册所有数据集"""
    try:
        from .data import datasets
        # 注册数据集
        for name in ['Pix3DDataset', 'CO3Dv2Dataset', 'ScanNetDataset', 
                     'KITTIDataset', 'VirtualKITTIDataset']:
            if hasattr(datasets, name):
                cls = getattr(datasets, name)
                dataset_name = name.replace('Dataset', '').lower()
                register('dataset', dataset_name)(cls)
    except ImportError as e:
        log.warning(f"Failed to auto-register datasets: {e}")