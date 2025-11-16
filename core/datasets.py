import os
import glob
import random
from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class RAFTColorDataset(Dataset):
    """
    RAFT 色彩遷移數據集 - SwinV2 專用版本
    只保留必要的數據處理邏輯
    """
    
    def __init__(self, video_data_root_list, image_size=[384, 512], 
                 min_frames=2, augment=True):
        # 處理多路徑輸入
        if isinstance(video_data_root_list, str):
            if ',' in video_data_root_list:
                self.video_data_root_list = [path.strip() for path in video_data_root_list.split(',')]
            else:
                self.video_data_root_list = [video_data_root_list]
        else:
            self.video_data_root_list = video_data_root_list
            
        self.image_size = image_size
        self.min_frames = min_frames
        self.augment = augment
        
        # ImageNet 標準化（SwinV2 用）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # ToTensor transform
        self.to_tensor = transforms.ToTensor()
        
        print(f"🎯 Loading RAFT Color Dataset (SwinV2 mode)")
        print(f"   Paths: {len(self.video_data_root_list)}")
        for i, path in enumerate(self.video_data_root_list, 1):
            print(f"   {i}. {path}")
            
        self.frame_pairs = self._load_frame_pairs()
        print(f"📊 Found {len(self.frame_pairs)} frame pairs")
        print(f"🎬 Scenes: {len(set(p['scene_id'] for p in self.frame_pairs))}")

    def _load_frame_pairs(self):
        """載入所有連續幀對"""
        all_frame_pairs = []
        
        for root_idx, video_data_root in enumerate(self.video_data_root_list):
            if not os.path.exists(video_data_root):
                print(f"⚠️ Path not found: {video_data_root}")
                continue
                
            pairs_count = 0
            for item in os.listdir(video_data_root):
                item_path = os.path.join(video_data_root, item)
                if os.path.isdir(item_path):
                    # 收集所有圖像
                    frames = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        frames.extend(glob.glob(os.path.join(item_path, ext)))
                    
                    frames = sorted(frames)
                    if len(frames) >= self.min_frames:
                        unique_scene_id = f"path{root_idx}_{item}"
                        
                        # 生成連續幀對
                        for i in range(len(frames) - 1):
                            all_frame_pairs.append({
                                'scene_id': unique_scene_id,
                                'frame1_path': frames[i],
                                'frame2_path': frames[i + 1],
                                'frame_idx': i,
                                'total_frames': len(frames)
                            })
                            pairs_count += 1
                            
            print(f"   ✅ Path {root_idx+1}: {pairs_count} pairs")

        random.shuffle(all_frame_pairs)
        return all_frame_pairs

    def _load_and_process_image(self, path):
        """
        載入並處理單張圖像
        
        Returns:
            rgb_gray: [3, H, W] - ImageNet標準化的灰階RGB（給SwinV2）
            lab: [3, H, W] - LAB色彩空間 (L:[0,100], ab:[-128,127])
        """
        try:
            # 載入並調整大小
            image = Image.open(path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            
            # ===== 1. 處理 RGB 灰階（給 SwinV2） =====
            image_gray = image.convert('L')
            gray_tensor = self.to_tensor(image_gray)  # [1, H, W] [0,1]
            rgb_gray = gray_tensor.repeat(3, 1, 1)    # [3, H, W]
            rgb_gray = self.normalize(rgb_gray)       # ImageNet標準化
            
            # ===== 2. 處理 LAB（給 Context 和色彩） =====
            image_np = np.array(image, dtype=np.uint8)
            lab_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # LAB 轉換到標準範圍
            lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0  # L: [0,100]
            lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0          # a: [-128,127]
            lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0          # b: [-128,127]
            
            lab = torch.from_numpy(lab_np).permute(2, 0, 1)  # [3, H, W]
            
            return rgb_gray, lab
            
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            # 返回默認值
            rgb_gray = torch.zeros(3, self.image_size[0], self.image_size[1])
            lab = torch.zeros(3, self.image_size[0], self.image_size[1])
            lab[0] = 50.0  # L通道設為中等亮度
            return rgb_gray, lab

    def _apply_augmentation(self, rgb1, rgb2, lab1, lab2):
        """
        同步數據增強
        
        Args:
            rgb1/rgb2: [3, H, W] - 已ImageNet標準化
            lab1/lab2: [3, H, W] - LAB格式
        """
        if not self.augment:
            return rgb1, rgb2, lab1, lab2
        
        # 水平翻轉（同步）
        if random.random() > 0.5:
            rgb1 = torch.flip(rgb1, [-1])
            rgb2 = torch.flip(rgb2, [-1])
            lab1 = torch.flip(lab1, [-1])
            lab2 = torch.flip(lab2, [-1])
        
        # 亮度調整（只對LAB的L通道）
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            lab1[0] = torch.clamp(lab1[0] * factor, 0, 100)
            lab2[0] = torch.clamp(lab2[0] * factor, 0, 100)
        
        # 飽和度調整（只對LAB的ab通道）
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            lab1[1:3] = torch.clamp(lab1[1:3] * factor, -128, 127)
            lab2[1:3] = torch.clamp(lab2[1:3] * factor, -128, 127)
        
        return rgb1, rgb2, lab1, lab2

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        pair_info = self.frame_pairs[idx]
        
        # 載入並處理兩幀
        rgb1, lab1 = self._load_and_process_image(pair_info['frame1_path'])
        rgb2, lab2 = self._load_and_process_image(pair_info['frame2_path'])
        
        # 數據增強（同步）
        rgb1, rgb2, lab1, lab2 = self._apply_augmentation(rgb1, rgb2, lab1, lab2)
        
        # ===== 準備輸出 =====
        # 1. Context（LAB 歸一化）
        L1 = lab1[0:1]  # [1, H, W]
        ab1 = lab1[1:3]  # [2, H, W]
        
        context1 = torch.cat([
            (L1 / 50.0) - 1.0,  # L: [0,100] → [-1,1]
            ab1 / 127.0          # ab: [-128,127] → [-1,1]
        ], dim=0)  # [3, H, W]
        
        # 2. 色彩通道（ab歸一化）
        ab1_norm = ab1 / 127.0       # [2, H, W] [-1,1]
        ab2_norm = lab2[1:3] / 127.0  # [2, H, W] [-1,1]
        
        # 3. Valid mask
        valid = torch.ones(self.image_size[0], self.image_size[1])
        
        return {
            # SwinV2 輸入
            'img1_rgb': rgb1,     # [3, H, W] ImageNet標準化的灰階RGB
            'img2_rgb': rgb2,     # [3, H, W] ImageNet標準化的灰階RGB
            
            # Context Encoder 輸入
            'context1': context1, # [3, H, W] LAB歸一化 [-1,1]
            
            # 色彩遷移數據
            'img1_ab': ab1_norm,  # [2, H, W] [-1,1]
            'img2_ab_gt': ab2_norm,  # [2, H, W] [-1,1]
            
            # 其他
            'valid': valid,       # [H, W]
            'scene_id': pair_info['scene_id'],
            'frame_idx': pair_info['frame_idx']
        }



