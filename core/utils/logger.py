# core/utils/logger.py - 增強版Logger

from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger
import time


class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg
        
        # ← 新增: 用於計算速度和ETA
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.last_print_step = 0

    def _print_training_status(self):
        """增強版訓練狀態顯示"""
        metrics_data = [self.running_loss[k]/self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        
        # ===== 計算速度和ETA =====
        current_time = time.time()
        elapsed = current_time - self.last_print_time
        steps_since_last = self.total_steps - self.last_print_step
        
        # 訓練速度 (steps/sec)
        if elapsed > 0:
            speed = steps_since_last / elapsed
        else:
            speed = 0.0
        
        # 進度百分比
        if hasattr(self.cfg.trainer, 'num_steps'):
            total_steps = self.cfg.trainer.num_steps
            progress = (self.total_steps / total_steps) * 100
            
            # 預估剩餘時間 (ETA)
            if speed > 0:
                remaining_steps = total_steps - self.total_steps
                eta_seconds = remaining_steps / speed
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_str = f"{eta_hours}h{eta_minutes:02d}m"
            else:
                eta_str = "N/A"
        else:
            progress = 0.0
            eta_str = "N/A"
        
        # ===== 格式化輸出 =====
        training_str = "[Step {:6d}/{:6d}] ".format(
            self.total_steps + 1, 
            total_steps if hasattr(self.cfg.trainer, 'num_steps') else 0
        )
        training_str += "[LR: {:.2e}] ".format(self.scheduler.get_last_lr()[0])
        training_str += "[Progress: {:5.1f}%] ".format(progress)
        training_str += "[Speed: {:4.1f} step/s] ".format(speed)
        training_str += "[ETA: {}] ".format(eta_str)
        
        # Loss metrics
        metrics_str = ""
        for k in sorted(self.running_loss.keys()):
            avg_val = self.running_loss[k] / self.cfg.sum_freq
            metrics_str += "{}: {:.4f}, ".format(k, avg_val)
        
        # print the training status
        loguru_logger.info(training_str + metrics_str)

        # ===== TensorBoard logging =====
        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.cfg.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0
        
        # 記錄速度和進度
        self.writer.add_scalar('speed_steps_per_sec', speed, self.total_steps)
        self.writer.add_scalar('progress_percent', progress, self.total_steps)
        
        # 更新時間記錄
        self.last_print_time = current_time
        self.last_print_step = self.total_steps

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        """關閉時顯示總訓練時間"""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        loguru_logger.info(f"Total training time: {hours}h {minutes}m")
        
        if self.writer is not None:
            self.writer.close()



