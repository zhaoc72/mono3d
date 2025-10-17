"""多GPU并行3DGS重建调度器"""
import torch
import torch.multiprocessing as mp
from queue import Queue
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)

class MultiGPUScheduler:
    """多GPU实例级并行调度器"""
    
    def __init__(self, num_gpus: int = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        self.num_gpus = num_gpus
        self.task_queues = [Queue() for _ in range(num_gpus)]
        self.result_queue = Queue()
        self.workers = []
        
        log.info(f"Initialized scheduler with {num_gpus} GPUs")
    
    def start(self):
        """启动工作进程"""
        mp.set_start_method('spawn', force=True)
        
        for gpu_id in range(self.num_gpus):
            worker = mp.Process(
                target=self._worker,
                args=(gpu_id, self.task_queues[gpu_id], self.result_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        log.info("All workers started")
    
    def submit(self, tasks: List[Dict[str, Any]]):
        """提交任务到队列"""
        for i, task in enumerate(tasks):
            gpu_id = i % self.num_gpus
            self.task_queues[gpu_id].put(task)
        
        # 发送终止信号
        for queue in self.task_queues:
            queue.put(None)
    
    def collect_results(self, num_tasks: int) -> List[Any]:
        """收集结果"""
        results = []
        for _ in range(num_tasks):
            result = self.result_queue.get()
            results.append(result)
        return results
    
    def stop(self):
        """停止所有工作进程"""
        for worker in self.workers:
            worker.join()
        log.info("All workers stopped")
    
    @staticmethod
    def _worker(gpu_id: int, task_queue: Queue, result_queue: Queue):
        """工作进程函数"""
        import torch
        from ..registry import build
        
        # 设置GPU
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        # 加载模型（每个进程独立加载）
        log.info(f"Worker {gpu_id}: Loading models...")
        gaussian_model = build("model", "gaussian", device=device)
        
        # 处理任务
        while True:
            task = task_queue.get()
            
            if task is None:
                break
            
            try:
                # 执行3DGS优化
                result = optimize_single_instance(
                    gaussian_model,
                    task['image'],
                    task['depth'],
                    task['mask'],
                    task['initial_shape'],
                    device=device
                )
                
                result_queue.put({
                    'task_id': task['task_id'],
                    'success': True,
                    'result': result,
                })
            
            except Exception as e:
                log.error(f"Worker {gpu_id}: Error processing task {task['task_id']}: {e}")
                result_queue.put({
                    'task_id': task['task_id'],
                    'success': False,
                    'error': str(e),
                })
        
        log.info(f"Worker {gpu_id}: Shutting down")


def optimize_single_instance(model, image, depth, mask, initial_shape, device):
    """单实例3DGS优化（在指定GPU上运行）"""
    
    # 将数据移到对应GPU
    image = image.to(device)
    depth = depth.to(device)
    mask = mask.to(device)
    
    # 初始化高斯
    model.initialize_from_shape(initial_shape)
    
    # 优化循环
    optimizer = torch.optim.Adam(model.parameters(), lr=1.6e-4)
    
    for iteration in range(3000):
        # 渲染
        rendered = model.render(camera=None)  # 简化示例
        
        # 计算损失
        loss = compute_loss(rendered, image, depth, mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 自适应密度控制
        if iteration % 100 == 0:
            model.densify()
            model.prune()
    
    # 返回结果
    return {
        'pointcloud': model.to_pointcloud().cpu(),
        'mesh': model.to_mesh().cpu(),
    }


def compute_loss(rendered, image, depth, mask):
    """计算损失（简化示例）"""
    color_loss = torch.nn.functional.mse_loss(rendered['color'], image)
    depth_loss = torch.nn.functional.l1_loss(rendered['depth'], depth)
    mask_loss = torch.nn.functional.binary_cross_entropy(rendered['alpha'], mask)
    
    return color_loss + 0.1 * depth_loss + 0.5 * mask_loss