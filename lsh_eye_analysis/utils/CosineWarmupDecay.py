import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupDecay(_LRScheduler):
    """
    带 Warmup 的余弦退火学习率调度器 (PyTorch Version)
    
    支持两种模式：
    1. 单次余弦退火（multi=0）：学习率从 initial_lr 平滑下降到 min_lr
    2. 重启式余弦退火（multi>0）：周期性重启，周期长度递增
    """

    def __init__(self, optimizer, initial_lr, min_lr, warmup_step, total_step, multi, print_step, last_epoch=-1, verbose=False):
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.warmup_step = warmup_step
        self.total_step = total_step
        self.multi = multi
        self.print_step = print_step
        
        # 内部状态初始化
        self.cycle_start_epoch = 0
        self.current_warmup_step = warmup_step
        self.current_total_step = total_step
        
        super(CosineWarmupDecay, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        rel_step = current_step - self.cycle_start_epoch
        
        # 检查周期重启
        if rel_step >= self.current_total_step:
            if self.multi > 0:
                # 周期递增
                self.current_warmup_step = int(self.current_warmup_step * (1.0 + self.multi))
                self.current_total_step = int(self.current_total_step * (1.0 + self.multi))
                # 重置相对步数，将当前步设为新周期的起点
                self.cycle_start_epoch = current_step
                rel_step = 0
        
        # 计算学习率
        lr = self.min_lr
        
        # 模式1: 单次且超过总步数 -> 保持 min_lr
        if self.multi == 0 and rel_step >= self.current_total_step:
            lr = self.min_lr
        else:
            # Warmup 阶段
            if rel_step < self.current_warmup_step:
                 # 线性增长: min_lr -> initial_lr
                 lr = self.min_lr + (self.initial_lr - self.min_lr) * (rel_step / self.current_warmup_step)
            else:
                # 余弦退火阶段
                cosine_step = rel_step - self.current_warmup_step
                cosine_steps_total = self.current_total_step - self.current_warmup_step
                
                if cosine_steps_total > 0:
                    lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                        (1.0 + math.cos(math.pi * cosine_step / cosine_steps_total))
                else:
                    lr = self.min_lr

        # 打印
        if self.print_step > 0 and current_step % self.print_step == 0:
            print(f"Step {current_step} - Learning rate: {lr:.8f}")

        return [lr for _ in self.base_lrs]