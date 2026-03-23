# train_colorization.py - MemFlowNet視頻色彩化訓練腳本 (無DDP簡化版)

from __future__ import print_function, division
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

# ← 按照原始MemFlowNet的import方式
import core.datasets_new as datasets  # 你修改後的dataloader
from core.loss_new import CompositeLoss, warp_color_by_flow  # 你的loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger
import random
from core.Networks import build_network
import os

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(gpu, cfg):
    """訓練主函數 (簡化版,無DDP)"""

    # ========== 設置GPU ==========
    rank = 0  # 單GPU,rank=0
    torch.cuda.set_device(gpu)

    # ========== 創建模型 ==========
    print("="*60)
    print(" Initializing MemFlowNet model...")
    print("="*60)

    model = build_network(cfg).cuda()  # ← 使用原始的build_network
    model = nn.DataParallel(model)  # ← 用DataParallel而非DDP
    model.train()

    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    # ========== 載入checkpoint ==========
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt

        # 过滤掉mask层（因为上采样倍数从8x改成2x，mask通道数不兼容）
        filtered_ckpt = {}
        for k, v in ckpt_model.items():
            # 更鲁棒的匹配：任何包含"mask"且以".weight"或".bias"结尾的层
            if 'mask' in k and ('.weight' in k or '.bias' in k):
                print(f"  ⚠️  Skipping incompatible layer: {k} (shape {v.shape})")
                continue
            filtered_ckpt[k] = v

        if 'module' in list(filtered_ckpt.keys())[0]:
            model.load_state_dict(filtered_ckpt, strict=False)
        else:
            model.module.load_state_dict(filtered_ckpt, strict=False)
        print("✅ Checkpoint loaded (mask layers skipped, will be randomly initialized)")

    # ========== 創建dataloader ==========
    print("\n" + "="*60)
    print(" Loading dataset...")
    print("="*60)

    train_loader = datasets.fetch_dataloader(cfg)  # ← 使用你修改的dataloader

    # ========== 創建optimizer ==========
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    # ========== 創建loss函數 ==========
    loss_fn = CompositeLoss(
        gamma=cfg.gamma,
        weight_l1=1.0,
        weight_perceptual=0.0,
        weight_contextual=0.0,
        weight_temporal=0.0,
        device='cuda'
    )

    # ========== 初始化 ==========
    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    epoch = 0
    if cfg.restore_steps > 1:
        print("[Loading optimizer from {}]".format(cfg.restore_ckpt))
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.total_steps = cfg.restore_steps - 1
        total_steps = cfg.restore_steps
        epoch = ckpt['epoch']
        for _ in range(total_steps):
            scheduler.step()

    # ========== 訓練循環 ==========
    print("\n" + "="*60)
    print(" Starting training...")
    print("="*60 + "\n")

    should_keep_training = True
    while should_keep_training:
        epoch += 1

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            # ===== 載入數據 =====
            images = data_blob['images'].cuda()            # [B, 4, 3, H, W] - LAB序列
            rgb_inputs = data_blob['rgb_inputs'].cuda()    # [B, 4, 3, H, W] - 灰階RGB（給 fnet）
            rgb_color_inputs = data_blob['rgb_color_inputs'].cuda()  # [B, 4, 3, H, W] - 彩色RGB（給 cnet）
            scene_ids = data_blob['scene_id']

            B, N, C, H, W = images.shape  # N=4

            # ===== LAB 歸一化到[-1, 1]，用於 warp / loss =====
            # L: [0,100] → [-1,1]
            # ab: [-128,127] → [-1,1]
            images_norm = images.clone()
            images_norm[:, :, 0] = (images[:, :, 0] / 50.0) - 1.0  # L
            images_norm[:, :, 1:3] = images[:, :, 1:3] / 127.0     # ab

            # ===== MemFlowNet Forward =====
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision, dtype=torch.bfloat16):
                # Encode context (前3幀) - 用彩色RGB提供顏色語境
                query, key, net, inp = model.module.encode_context(rgb_color_inputs[:, :-1, ...])

                # Encode features (所有4幀) - 用灰階RGB讓twins對齊預訓練分布
                coords0, coords1, fmaps = model.module.encode_features(rgb_inputs)

                # 逐幀預測
                values = None
                all_color_predictions = []

                for ti in range(cfg.input_frames - 1):  # 0,1,2
                    # Memory管理
                    if ti < cfg.num_ref_frames:
                        ref_values = values
                        ref_keys = key[:, :, :ti + 1] if ti > 0 else key[:, :, :1]
                    else:
                        indices = [torch.randperm(ti)[:cfg.num_ref_frames - 1] for _ in range(B)]
                        ref_values = torch.stack([
                            values[bi, :, indices[bi]] for bi in range(B)
                        ], 0)
                        ref_keys = torch.stack([
                            key[bi, :, indices[bi]] for bi in range(B)
                        ], 0)
                        ref_keys = torch.cat([ref_keys, key[:, :, ti].unsqueeze(2)], dim=2)

                    # Predict flow
                    flow_pr, current_value, conf = model.module.predict_flow(
                        net[:, ti],
                        inp[:, ti],
                        coords0,
                        coords1,
                        fmaps[:, ti:ti + 2],
                        query[:, :, ti],
                        ref_keys,
                        ref_values
                    )
                    # flow_pr: list of [B, 2, H/8, W/8]

                    # 累積memory
                    values = current_value if values is None else torch.cat([values, current_value], dim=2)

                    # Color warping (修改为H/4分辨率warping)
                    source_ab = images_norm[:, ti, 1:3]  # [B, 2, H, W]

                    color_preds_ti = []
                    for flow_pred in flow_pr:
                        # 现在flow是H/4分辨率 (2倍上采样的结果)
                        flow_h, flow_w = flow_pred.shape[2:]

                        # 下采样source_ab到与flow相同的分辨率
                        if flow_h != H or flow_w != W:
                            source_ab_resized = nn.functional.interpolate(
                                source_ab,
                                size=(flow_h, flow_w),
                                mode='bilinear',
                                align_corners=True
                            )
                        else:
                            source_ab_resized = source_ab

                        # 在flow的分辨率下warp color
                        warped_color = warp_color_by_flow(source_ab_resized, flow_pred)
                        color_preds_ti.append(warped_color)

                    all_color_predictions.append(color_preds_ti)

            # ===== 計算Loss =====
            total_loss = 0.0
            metrics = {'color_error': 0.0, 'epe': 0.0}

            for ti in range(cfg.input_frames - 1):
                gt_ab = images_norm[:, ti + 1, 1:3]  # [B, 2, H, W]
                L_channel = images[:, ti + 1, 0:1]  # [B, 1, H, W] [0,100]

                color_preds = all_color_predictions[ti]

                # 下采样GT到与预测相同的分辨率 (H/4)
                pred_h, pred_w = color_preds[0].shape[2:]
                if pred_h != H or pred_w != W:
                    gt_ab = nn.functional.interpolate(
                        gt_ab,
                        size=(pred_h, pred_w),
                        mode='bilinear',
                        align_corners=True
                    )
                    L_channel = nn.functional.interpolate(
                        L_channel,
                        size=(pred_h, pred_w),
                        mode='bilinear',
                        align_corners=True
                    )

                loss_ti, loss_dict_ti = loss_fn(
                    color_preds,
                    gt_ab,
                    L_channel,
                    last_ab_pred=None,
                    flow_forward=None
                )

                total_loss += loss_ti
                for key in loss_dict_ti:
                    if key not in metrics:
                        metrics[key] = 0.0
                    if isinstance(loss_dict_ti[key], torch.Tensor):
                        metrics[key] += loss_dict_ti[key].item()
                    else:
                        metrics[key] += loss_dict_ti[key]

            # 平均
            total_loss = total_loss / (cfg.input_frames - 1)
            for key in metrics:
                metrics[key] = metrics[key] / (cfg.input_frames - 1)

            metrics['total_loss'] = total_loss.item()

            # ===== Backward =====
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # ===== Logging =====
            if rank == 0:
                logger.push(metrics)

            # ===== Checkpoint =====
            if total_steps % cfg.val_freq == cfg.val_freq - 1 and rank == 0:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1, cfg.name)
                torch.save({
                    'iteration': total_steps,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'model': model.module.state_dict(),
                }, PATH)
                print(f"\n Checkpoint saved: {PATH}\n")

            total_steps += 1

            if total_steps >= cfg.trainer.num_steps:
                should_keep_training = False
                break

    # ===== 訓練結束 =====
    logger.close()
    if rank == 0:
        PATH = cfg.log_dir + f'/{cfg.name}.pth'
        torch.save(model.module.state_dict(), PATH)
        print(f" Final model saved: {PATH}")

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='memflow_colorization', help="name your experiment")
    parser.add_argument('--stage', default='colorization', help="stage name")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--GPU_ids', type=str, default='0')

    args = parser.parse_args()

    # ========== 創建config (模仿原始方式) ==========
    from configs.colorization_memflownet import get_cfg  # ← 需要創建這個config文件

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)

    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    # 設置隨機種子
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    # 開始訓練 (單GPU,gpu=0)
    train(gpu=0, cfg=cfg)
