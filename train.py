from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT

# 導入你的 Dataset
from datasets import RAFTColorDataset

# 🔥 導入新的loss模組
from loss import CompositeLoss

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self, enabled=True):
            self._enabled = enabled
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# 常數設定
SUM_FREQ = 10000
VAL_FREQ = 5000

def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """創建優化器和學習率調度器"""
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.wdecay, 
        eps=args.epsilon
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        args.lr, 
        args.num_steps + 100,
        pct_start=0.05, 
        cycle_momentum=False, 
        anneal_strategy='linear'
    )

    return optimizer, scheduler

def fetch_dataloader(args):
    """創建數據加載器"""
    
    # 處理數據路徑
    if hasattr(args, 'data_path') and args.data_path:
        if ',' in args.data_path:
            video_paths = [path.strip() for path in args.data_path.split(',')]
        else:
            video_paths = [args.data_path]
    else:
        raise ValueError("Please provide --data_path argument")
    
    print(f"📁 Loading data from {len(video_paths)} paths:")
    for i, path in enumerate(video_paths, 1):
        print(f"   {i}. {path}")

    # 創建數據集
    train_dataset = RAFTColorDataset(
        video_data_root_list=video_paths,
        image_size=args.image_size,
        augment=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=False, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True
    )

    print(f'✅ Training with {len(train_dataset)} frame pairs')
    return train_loader

class Logger:
    """簡單的日誌記錄器"""
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, 
            self.scheduler.get_last_lr()[0]
        )
        metrics_str = ("{:10.6f}, " * len(metrics_data)).format(*metrics_data)
        
        print(training_str + metrics_str)

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        pass

    def close(self):
        pass

def train(args):
    """主訓練函數"""
    
    # 🔥 創建模型
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print(f"✅ Model created")
    print(f"📊 Parameter Count: {count_parameters(model):,}")

    # 🔥 載入 checkpoint(如果有)
    if args.restore_ckpt is not None:
        print(f"📥 Loading checkpoint: {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Checkpoint loaded (strict=False)")

    model.cuda()
    model.train()

    # 凍結 BN
    if args.stage != 'chairs':
        model.module.freeze_bn()
        print("❄️ BatchNorm frozen")

    # 🔥 創建數據加載器
    train_loader = fetch_dataloader(args)
    
    # 🔥 創建優化器
    optimizer, scheduler = fetch_optimizer(args, model)

    # 🔥 創建loss函數 (新增)
    loss_fn = CompositeLoss(
        gamma=args.gamma,
        weight_l1=args.weight_l1,
        weight_perceptual=args.weight_perceptual,
        weight_contextual=args.weight_contextual,
        weight_temporal=args.weight_temporal,
        device='cuda'
    )

    # 初始化
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    # 🔥 用於temporal loss的上一幀預測緩存
    prev_ab_pred = None
    prev_scene_id = None

    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60 + "\n")

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 🔥 載入數據
            img1_rgb = data_blob['img1_rgb'].cuda()      # [B, 3, H, W] ImageNet標準化
            img2_rgb = data_blob['img2_rgb'].cuda()      # [B, 3, H, W] ImageNet標準化
            context1 = data_blob['context1'].cuda()      # [B, 3, H, W] LAB歸一化
            
            img1_ab = data_blob['img1_ab'].cuda()        # [B, 2, H, W] [-1,1]
            img2_ab_gt = data_blob['img2_ab_gt'].cuda()  # [B, 2, H, W] [-1,1]
            
            scene_ids = data_blob['scene_id']            # 場景ID列表

            # 🔥 提取L通道 (從context1恢復)
            # context1 = [(L/50)-1, ab/127], 需要恢復L通道
            L_channel = (context1[:, 0:1, :, :] + 1.0) * 50.0  # [B, 1, H, W] [0,100]

            # 🔥 RAFT 前向傳播
            flow_predictions = model(
                img1_rgb, 
                img2_rgb, 
                context1, 
                iters=args.iters
            )

            # 🔥 使用光流進行色彩遷移
            color_predictions = []
            for flow_pred in flow_predictions:
                # 使用loss.py中的warp函數
                from loss import warp_color_by_flow
                warped_color = warp_color_by_flow(img1_ab, flow_pred)
                color_predictions.append(warped_color)

            # 🔥 準備temporal loss的數據
            # 檢查當前batch中的場景是否與上一次相同
            current_scene_id = scene_ids[0]  # 假設batch內場景相同
            
            # 如果場景連續,使用上一幀預測;否則設為None
            use_temporal = prev_ab_pred is not None and current_scene_id == prev_scene_id
            
            if use_temporal:
                # 確保尺寸匹配
                if prev_ab_pred.shape == img2_ab_gt.shape:
                    last_ab_pred = prev_ab_pred
                    flow_forward = flow_predictions[-1]  # 使用最終的flow
                else:
                    last_ab_pred = None
                    flow_forward = None
            else:
                last_ab_pred = None
                flow_forward = None

            # 🔥 計算loss (使用新的CompositeLoss)
            loss, loss_dict = loss_fn(
                color_predictions,   # 色彩預測序列
                img2_ab_gt,         # Ground truth
                L_channel,          # L通道
                last_ab_pred,       # 上一幀預測 (可能為None)
                flow_forward        # 光流 (可能為None)
            )
            
            # 🔥 更新上一幀預測 (用於下一次迭代)
            with torch.no_grad():
                prev_ab_pred = color_predictions[-1].detach().clone()
                prev_scene_id = current_scene_id
            
            # 反向傳播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # 🔥 記錄 (只記錄數值metrics)
            metrics_to_log = {}
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    metrics_to_log[key] = val.item()
                else:
                    metrics_to_log[key] = val
            
            logger.push(metrics_to_log)

            # 定期保存
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                #print(f"💾 Checkpoint saved: {PATH}")

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            # 檢查是否完成訓練
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    # 最終保存
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    print(f"\n✅ Training completed!")
    print(f"💾 Final model saved: {PATH}")

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 基礎參數
    parser.add_argument('--name', default='raft_swinv2', help="experiment name")
    parser.add_argument('--stage', default='chairs', help="training stage") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')

    # 訓練參數
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # RAFT 參數
    parser.add_argument('--iters', type=int, default=9)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    
    # 🔥 Loss權重參數 (新增)
    parser.add_argument('--weight_l1', type=float, default=1.0, help='L1 loss weight')
    parser.add_argument('--weight_perceptual', type=float, default=0.6, help='Perceptual loss weight')
    # 0.6
    parser.add_argument('--weight_contextual', type=float, default=0.3, help='Contextual loss weight')
    #0.3
    parser.add_argument('--weight_temporal', type=float, default=0.2, help='Temporal consistency loss weight')
    #0.2
    
    # 數據路徑參數
    parser.add_argument('--data_path', type=str, required=True, 
                       help='path to video data (comma-separated for multiple paths)')
    
    args = parser.parse_args()

    # 設定隨機種子
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 創建 checkpoint 目錄
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # 開始訓練
    train(args)



