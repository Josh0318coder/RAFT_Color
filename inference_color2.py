# inference_colorization.py - MemFlowNet視頻色彩化推理腳本 (Autoregressive)

from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # 非交互式後端
import matplotlib.pyplot as plt
from matplotlib import cm

from core.Networks import build_network
from core.loss_new import warp_color_by_flow
from loguru import logger as loguru_logger


class ColorizationInferenceCore:
    """色彩化推理核心 - 帶memory管理"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.clear_memory()
        
    def clear_memory(self):
        """清空memory (每個新視頻開始時調用)"""
        self.curr_ti = -1
        self.keys = None    # Store historical keys
        self.values = None  # Store historical values
        
    def step(self, images_norm, query, key, net, inp, coords0, coords1, fmaps):
        """
        處理一幀
        
        Args:
            images_norm: [1, 2, 3, H, W] - 當前幀和下一幀的LAB歸一化
            query, key, net, inp: context encoding結果
            coords0, coords1: 初始坐標
            fmaps: 特徵圖
            
        Returns:
            flow_final: [1, 2, H/8, W/8] - flow預測
            current_value: memory value
        """
        self.curr_ti += 1
        B = images_norm.shape[0]
        
        # Memory管理
        if self.curr_ti == 0:
            # 第一幀: 沒有歷史
            ref_values = None
            ref_keys = key.unsqueeze(2)  # [B, C, 1, H, W]
        elif self.curr_ti < self.config.num_ref_frames:
            # 前幾幀: 使用所有歷史
            ref_values = self.values
            # 累積所有歷史keys + 當前key
            all_keys = []
            for ti in range(self.curr_ti):
                all_keys.append(self.keys[:, :, ti:ti+1])
            all_keys.append(key.unsqueeze(2))
            ref_keys = torch.cat(all_keys, dim=2)
        else:
            # 歷史過多: 隨機採樣
            indices = [torch.randperm(self.curr_ti)[:self.config.num_ref_frames - 1] for _ in range(B)]
            ref_values = torch.stack([
                self.values[bi, :, indices[bi]] for bi in range(B)
            ], 0)
            ref_keys = torch.stack([
                self.keys[bi, :, indices[bi]] for bi in range(B)
            ], 0)
            ref_keys = torch.cat([ref_keys, key.unsqueeze(2)], dim=2)
        
        # Predict flow
        flow_predictions, current_value, confidence_map = self.model.predict_flow(
            net,
            inp,
            coords0,
            coords1,
            fmaps,
            query.unsqueeze(2),  # 加時間維度 [B, C, 1, H, W]
            ref_keys,
            ref_values
        )

        # 取最後一次迭代的結果
        flow_final = flow_predictions[-1]

        # 累積memory (store keys and values separately)
        if self.keys is None:
            self.keys = key.unsqueeze(2)
            self.values = current_value
        else:
            self.keys = torch.cat([self.keys, key.unsqueeze(2)], dim=2)
            self.values = torch.cat([self.values, current_value], dim=2)

        return flow_final, current_value, confidence_map


_imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
_to_tensor = transforms.ToTensor()


def load_image_as_lab(image_path, target_size=(224, 224)):
    """
    載入圖像並轉換為LAB格式，同時回傳 ImageNet normalize 的灰階 RGB

    Args:
        image_path: 圖像路徑
        target_size: (H, W) 目標尺寸

    Returns:
        lab: [3, H, W] tensor, LAB格式
        rgb_gray: [3, H, W] tensor, 灰階重複3通道並 ImageNet normalize（給 encoder 用）
        original_size: (W, H) 原始尺寸
    """
    # 載入圖像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)

    # Resize到目標尺寸
    image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)

    # ===== 灰階 RGB (給 encoder) =====
    image_gray = image.convert('L')
    gray_tensor = _to_tensor(image_gray)          # [1, H, W]
    rgb_gray = gray_tensor.repeat(3, 1, 1)        # [3, H, W]
    rgb_gray = _imagenet_normalize(rgb_gray)      # ImageNet 標準化

    # ===== LAB (給 warp / loss) =====
    image_np = np.array(image, dtype=np.uint8)
    image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 標準化範圍
    image_lab[:, :, 0] = image_lab[:, :, 0] * 100.0 / 255.0  # L: [0,100]
    image_lab[:, :, 1] = image_lab[:, :, 1] - 128.0          # a: [-128,127]
    image_lab[:, :, 2] = image_lab[:, :, 2] - 128.0          # b: [-128,127]

    lab = torch.from_numpy(image_lab).permute(2, 0, 1)  # [3, H, W]

    return lab, rgb_gray, original_size


def lab_to_rgb(lab_tensor):
    """
    LAB tensor轉RGB numpy
    
    Args:
        lab_tensor: [3, H, W] tensor, LAB格式
        
    Returns:
        rgb_np: [H, W, 3] numpy array, RGB格式
    """
    lab_np = lab_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 轉回OpenCV LAB格式
    lab_cv = lab_np.copy()
    lab_cv[:, :, 0] = lab_np[:, :, 0] * 255.0 / 100.0  # L
    lab_cv[:, :, 1] = lab_np[:, :, 1] + 128.0          # a
    lab_cv[:, :, 2] = lab_np[:, :, 2] + 128.0          # b
    
    lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)
    
    # 轉RGB
    bgr_np = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    
    return rgb_np


def lab_tensor_to_rgb_normalized(lab_tensor):
    """
    LAB tensor 轉 ImageNet normalized 彩色 RGB tensor（給 cnet 用）

    Args:
        lab_tensor: [3, H, W] tensor, LAB格式 (L:[0,100], ab:[-128,127])

    Returns:
        rgb_normalized: [3, H, W] tensor, ImageNet normalized
    """
    rgb_np = lab_to_rgb(lab_tensor)                      # [H, W, 3] uint8
    rgb_pil = Image.fromarray(rgb_np)
    rgb_tensor = _to_tensor(rgb_pil)                     # [3, H, W] in [0,1]
    return _imagenet_normalize(rgb_tensor)               # [3, H, W]


def visualize_confidence_map(confidence_np, cmap='viridis'):
    """
    將 confidence map 轉換為視覺化的彩色圖像

    Args:
        confidence_np: [H, W] numpy array, 值域 [0, 1]
        cmap: matplotlib colormap 名稱 ('viridis', 'jet', 'hot', 'plasma' 等)

    Returns:
        vis_image: [H, W, 3] numpy array, RGB格式, 值域 [0, 255]
    """
    # 確保值域在 [0, 1]
    confidence_np = np.clip(confidence_np, 0, 1)

    # 使用 matplotlib colormap 將值映射到顏色
    colormap = cm.get_cmap(cmap)
    colored = colormap(confidence_np)  # [H, W, 4] (RGBA)

    # 轉換為 RGB (去掉 alpha 通道) 並縮放到 [0, 255]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


@torch.no_grad()
def colorize_video(model, video_frames, processor, target_size=(224, 224)):
    """
    對一個視頻序列進行色彩化 (Autoregressive推理)

    Args:
        model: MemFlowNet模型
        video_frames: list of frame paths
        processor: ColorizationInferenceCore實例
        target_size: (H, W)

    Returns:
        colorized_frames: list of RGB numpy arrays
        confidence_maps: list of confidence numpy arrays
    """
    processor.clear_memory()  # 清空memory

    colorized_frames = []
    confidence_maps = []

    # ===== 第一幀: 使用GT (reference frame) =====
    first_lab, _, _ = load_image_as_lab(video_frames[0], target_size)
    first_rgb = lab_to_rgb(first_lab)
    colorized_frames.append(first_rgb)
    # 第一幀沒有 confidence（因為是 GT）
    confidence_maps.append(None)

    # 保存第一幀的AB作為起始
    last_predicted_ab_norm = first_lab[1:3] / 127.0  # 歸一化到[-1, 1]

    # cnet 彩色 RGB 輸入：從第一幀 GT LAB 初始化
    last_cnet_rgb = lab_tensor_to_rgb_normalized(first_lab)  # [3, H, W]

    # ===== 從第二幀開始autoregressive推理 =====
    for i in tqdm(range(len(video_frames) - 1), desc="  Colorizing", leave=False):
        # 載入當前幀和下一幀
        frame_t_lab, frame_t_rgb_gray, _ = load_image_as_lab(video_frames[i], target_size)
        frame_t1_lab, frame_t1_rgb_gray, _ = load_image_as_lab(video_frames[i+1], target_size)

        # ===== 灰階 RGB 輸入 (給 encoder，對齊預訓練分布) =====
        rgb_inputs = torch.stack([frame_t_rgb_gray, frame_t1_rgb_gray], dim=0).unsqueeze(0).cuda()  # [1, 2, 3, H, W]

        # ===== LAB 輸入 (給 warp / loss) =====
        frame_t_norm = torch.zeros_like(frame_t_lab)
        frame_t1_norm = torch.zeros_like(frame_t1_lab)

        # L通道: 使用GT
        frame_t_norm[0] = (frame_t_lab[0] / 50.0) - 1.0
        frame_t1_norm[0] = (frame_t1_lab[0] / 50.0) - 1.0

        # AB通道: 使用上一幀的預測結果 (autoregressive!)
        frame_t_norm[1:3] = last_predicted_ab_norm
        frame_t1_norm[1:3] = last_predicted_ab_norm  # 初始猜測,會被更新

        # Stack成[1, 2, 3, H, W]
        images_norm = torch.stack([frame_t_norm, frame_t1_norm], dim=0).unsqueeze(0).cuda()

        H, W = target_size

        # Forward pass
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # Encode context - 用前一幀的預測彩色 RGB
            cnet_input = last_cnet_rgb.unsqueeze(0).cuda()  # [1, 3, H, W]
            query, key, net, inp = model.encode_context(cnet_input)

            # Encode features - 用灰階RGB對齊預訓練分布
            coords0, coords1, fmaps = model.encode_features(rgb_inputs)

            # Predict flow
            flow_final, current_value, confidence_map = processor.step(
                images_norm, query, key, net, inp, coords0, coords1, fmaps
            )
        
        # 现在flow是H/4分辨率 (2倍上采样的结果)
        flow_h, flow_w = flow_final.shape[2:]

        # 下采样source_ab到与flow相同的分辨率
        source_ab = last_predicted_ab_norm.unsqueeze(0).cuda()  # [1, 2, H, W]
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
        warped_ab = warp_color_by_flow(source_ab_resized, flow_final)

        # 上采样回H分辨率
        if flow_h != H or flow_w != W:
            warped_ab = nn.functional.interpolate(
                warped_ab,
                size=(H, W),
                mode='bilinear',
                align_corners=True
            )
        
        # 更新last_predicted_ab_norm (用於下一次迭代)
        last_predicted_ab_norm = warped_ab.squeeze(0).cpu()

        # 組合L + warped AB
        colorized_lab = torch.cat([
            frame_t1_lab[0:1],  # 使用目標幀的L通道 (GT)
            last_predicted_ab_norm * 127.0  # 預測的AB通道
        ], dim=0)

        # 更新 cnet 彩色 RGB 輸入（下一幀使用）
        last_cnet_rgb = lab_tensor_to_rgb_normalized(colorized_lab)

        # 轉RGB
        colorized_rgb = lab_to_rgb(colorized_lab)
        colorized_frames.append(colorized_rgb)

        # 保存 confidence map (轉為 numpy array)
        confidence_np = confidence_map.squeeze().cpu().numpy()  # [H, W]
        confidence_maps.append(confidence_np)

    return colorized_frames, confidence_maps


def process_video_directory(input_dir, output_dir, model, config, target_size=(224, 224)):
    """
    處理一個視頻文件夾
    
    Args:
        input_dir: 輸入視頻文件夾路徑
        output_dir: 輸出文件夾路徑
        model: MemFlowNet模型
        config: 配置
        target_size: (H, W)
    """
    # 獲取所有圖像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    frames = []
    for ext in image_extensions:
        frames.extend(glob.glob(os.path.join(input_dir, ext)))
        frames.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    frames = sorted(frames)
    
    if len(frames) < 2:
        print(f"  ⚠️ Skipping {input_dir}: need at least 2 frames")
        return
    
    print(f"  📹 Found {len(frames)} frames")
    
    # 創建inference processor
    processor = ColorizationInferenceCore(model, config)
    
    # 色彩化
    colorized_frames, confidence_maps = colorize_video(model, frames, processor, target_size)

    # 保存結果
    os.makedirs(output_dir, exist_ok=True)
    confidence_dir = os.path.join(output_dir, 'confidence')
    os.makedirs(confidence_dir, exist_ok=True)

    for i, (frame_path, colorized, confidence) in enumerate(zip(frames, colorized_frames, confidence_maps)):
        # 保持原始文件名
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, frame_name)

        # 保存彩色圖像
        Image.fromarray(colorized).save(output_path)

        # 保存 confidence map 視覺化圖像（如果不是 None）
        if confidence is not None:
            # 將 confidence 轉換為視覺化圖像
            conf_vis = visualize_confidence_map(confidence, cmap='viridis')

            # 保存為 PNG 圖片
            conf_name = os.path.splitext(frame_name)[0] + '_confidence.png'
            conf_path = os.path.join(confidence_dir, conf_name)
            Image.fromarray(conf_vis).save(conf_path)

    print(f"  ✅ Saved {len(colorized_frames)} frames to {output_dir}")
    print(f"  ✅ Saved {len([c for c in confidence_maps if c is not None])} confidence maps to {confidence_dir}")


def main():
    parser = argparse.ArgumentParser(description='MemFlowNet視頻色彩化推理 (Autoregressive)')
    
    # 基本參數
    parser.add_argument('--input_dir', required=True, help='輸入數據根目錄')
    parser.add_argument('--output_dir', required=True, help='輸出數據根目錄')
    parser.add_argument('--checkpoint', required=True, help='訓練好的checkpoint路徑')
    parser.add_argument('--stage', default='colorization', help='stage name')
    
    # 模型參數
    parser.add_argument('--image_size', type=int, default=224, help='處理尺寸')
    parser.add_argument('--GPU_ids', type=str, default='0', help='GPU ID')
    
    args = parser.parse_args()
    
    # 設置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    
    # 載入配置
    print("="*60)
    print("🔧 Loading configuration...")
    print("="*60)
    
    from configs.colorization_memflownet import get_cfg
    cfg = get_cfg()
    cfg.restore_ckpt = args.checkpoint
    
    # 載入模型
    print("\n" + "="*60)
    print("🔧 Loading model...")
    print("="*60)
    
    model = build_network(cfg).cuda()
    model = nn.DataParallel(model)
    
    # 載入checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
    
    if 'module' in list(ckpt_model.keys())[0]:
        model.load_state_dict(ckpt_model, strict=False)
    else:
        model.module.load_state_dict(ckpt_model, strict=False)
    
    model.eval()
    print("✅ Model loaded\n")
    
    # 掃描輸入目錄
    print("="*60)
    print(f"📂 Scanning input directory: {args.input_dir}")
    print("="*60)
    
    video_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            video_dirs.append((item, item_path))
    
    video_dirs = sorted(video_dirs)
    print(f"📊 Found {len(video_dirs)} video directories\n")
    
    # 處理每個視頻
    target_size = (args.image_size, args.image_size)
    
    for video_name, video_path in video_dirs:
        print(f"🎬 Processing: {video_name}")
        
        output_video_dir = os.path.join(args.output_dir, video_name)
        
        try:
            process_video_directory(
                video_path, 
                output_video_dir, 
                model.module, 
                cfg, 
                target_size
            )
        except Exception as e:
            print(f"  ❌ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("✅ All videos processed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()



