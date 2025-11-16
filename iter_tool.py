#!/usr/bin/env python3
"""
RAFT迭代收斂分析工具
統計每個iteration階段的L1 loss，用於判斷最優迭代次數
支援多數據集路徑，輸出詳細統計報告和可視化圖表
"""

import sys
sys.path.append('core')

import os
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import defaultdict

from raft import RAFT
from utils.utils import InputPadder

# 設備配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class IterationAnalyzer:
    """RAFT迭代分析器"""
    
    def __init__(self, model_path, small_model=False, image_size=[224, 224], max_iters=12):
        self.image_size = image_size
        self.max_iters = max_iters
        
        # ImageNet 標準化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 載入模型
        print(f"🔧 載入模型: {model_path}")
        self.model = self.load_model(model_path, small_model)
        print(f"✅ 模型已載入到 {DEVICE}")
        
        # 統計數據存儲
        self.all_losses = []  # 每個樣本的loss序列
        self.scene_losses = defaultdict(list)  # 按場景分組
        self.flow_magnitude_losses = defaultdict(list)  # 按光流幅度分組
        
    def load_model(self, model_path, small_model):
        """載入RAFT模型"""
        args = argparse.Namespace()
        args.small = small_model
        args.mixed_precision = False
        args.alternate_corr = False
        
        model = RAFT(args)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 處理checkpoint格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 移除'module.'前綴
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_checkpoint[k[7:]] = v
            else:
                new_checkpoint[k] = v
        
        model.load_state_dict(new_checkpoint, strict=False)
        model = model.to(DEVICE)
        model.eval()
        return model
    
    def find_frame_pairs(self, input_paths):
        """
        查找所有可用的幀對
        
        Args:
            input_paths: list of str - 數據集路徑列表
        
        Returns:
            frame_pairs: list of dict - 所有幀對信息
        """
        print(f"🔍 掃描數據集...")
        print(f"   路徑數量: {len(input_paths)}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        frame_pairs = []
        
        for path_idx, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                print(f"⚠️ 路徑不存在: {input_path}")
                continue
            
            print(f"   [{path_idx+1}/{len(input_paths)}] 掃描: {input_path}")
            pairs_count = 0
            
            # 遍歷子文件夾（場景）
            for scene_dir in os.listdir(input_path):
                scene_path = os.path.join(input_path, scene_dir)
                if os.path.isdir(scene_path):
                    # 獲取該場景的所有幀
                    frames = []
                    for ext in image_extensions:
                        frames.extend(glob.glob(os.path.join(scene_path, f'*{ext}')))
                        frames.extend(glob.glob(os.path.join(scene_path, f'*{ext.upper()}')))
                    
                    frames = sorted(frames)
                    
                    # 生成連續幀對
                    for i in range(len(frames) - 1):
                        frame_pairs.append({
                            'dataset_idx': path_idx,
                            'dataset_path': input_path,
                            'scene': scene_dir,
                            'scene_id': f"dataset{path_idx}_{scene_dir}",
                            'frame1_path': frames[i],
                            'frame2_path': frames[i + 1],
                            'frame1_name': os.path.basename(frames[i]),
                            'frame2_name': os.path.basename(frames[i + 1]),
                            'frame_idx': i
                        })
                        pairs_count += 1
            
            print(f"      ✓ 找到 {pairs_count} 個幀對")
        
        print(f"\n📊 總計: {len(frame_pairs)} 個幀對")
        print(f"🎬 場景數: {len(set(p['scene_id'] for p in frame_pairs))}")
        return frame_pairs
    
    def load_image_as_lab(self, image_path):
        """載入圖像並轉換為LAB格式"""
        try:
            # 載入並調整尺寸
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            image_np = np.array(image, dtype=np.uint8)
            
            # 轉換到LAB色彩空間
            image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            image_lab = image_lab.astype(np.float32)
            
            # OpenCV LAB格式轉換為標準LAB範圍
            image_lab[:, :, 0] = image_lab[:, :, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
            image_lab[:, :, 1] = image_lab[:, :, 1] - 128.0          # A: [0,255] -> [-128,127]
            image_lab[:, :, 2] = image_lab[:, :, 2] - 128.0          # B: [0,255] -> [-128,127]
            
            # 轉為tensor [3, H, W]
            image_lab = torch.from_numpy(image_lab).permute(2, 0, 1)
            return image_lab
            
        except Exception as e:
            print(f"❌ 載入圖像失敗 {image_path}: {e}")
            return None
    
    def prepare_raft_inputs(self, img_lab):
        """準備RAFT輸入"""
        # 分離L和ab通道
        img_L = img_lab[0:1]  # [1, H, W] [0, 100]
        img_ab = img_lab[1:3]  # [2, H, W] [-128, 127]
        
        # L通道歸一化到[0,1]，再做ImageNet標準化
        img_L_norm = img_L / 100.0
        img_input = img_L_norm.repeat(3, 1, 1)
        img_input = self.normalize(img_input)
        
        # Context輸入: 完整LAB歸一化到[-1, 1]
        img_L_context = (img_L / 50.0) - 1.0
        img_ab_norm_context = img_ab / 127.0
        img_context_input = torch.cat([img_L_context, img_ab_norm_context], dim=0)
        
        # ab通道歸一化到[-1, 1]
        img_ab_norm = img_ab / 127.0
        
        return img_input, img_context_input, img_L, img_ab_norm
    
    def warp_color_by_flow(self, img1_ab, flow):
        """使用光流進行色彩傳播"""
        B, _, H, W = flow.shape
        
        # 創建坐標網格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=flow.dtype),
            torch.arange(W, device=flow.device, dtype=flow.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # 應用光流
        flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        sampling_coords = base_grid + flow_permuted
        
        # 歸一化到[-1, 1]
        sampling_coords[..., 0] = 2.0 * sampling_coords[..., 0] / (W - 1) - 1.0
        sampling_coords[..., 1] = 2.0 * sampling_coords[..., 1] / (H - 1) - 1.0
        
        # 雙線性插值採樣
        warped_color = F.grid_sample(
            img1_ab, sampling_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return torch.clamp(warped_color, -1, 1)
    
    def analyze_frame_pair(self, frame_pair):
        """
        分析單個幀對，記錄每次迭代的loss
        
        Returns:
            result: dict - 包含各次迭代的loss和元信息
        """
        # 載入兩幀
        img1_lab = self.load_image_as_lab(frame_pair['frame1_path'])
        img2_lab = self.load_image_as_lab(frame_pair['frame2_path'])
        
        if img1_lab is None or img2_lab is None:
            return None
        
        # 準備輸入
        img1_input, img1_context_input, img1_L, img1_ab_norm = self.prepare_raft_inputs(img1_lab)
        img2_input, img2_context_input, img2_L, img2_ab_norm = self.prepare_raft_inputs(img2_lab)
        
        # GT色彩
        gt_ab = img2_ab_norm  # [2, H, W] [-1, 1]
        
        with torch.no_grad():
            # 轉為GPU tensor
            img1_tensor = img1_input.unsqueeze(0).to(DEVICE)
            img2_tensor = img2_input.unsqueeze(0).to(DEVICE)
            img1_context_tensor = img1_context_input.unsqueeze(0).to(DEVICE)
            img1_ab_tensor = img1_ab_norm.unsqueeze(0).to(DEVICE)
            gt_ab_tensor = gt_ab.unsqueeze(0).to(DEVICE)
            
            # Padding
            padder = InputPadder(img1_tensor.shape)
            img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)
            img1_context_padded = padder.pad(img1_context_tensor)[0]
            
            # 🔥 關鍵: RAFT推理，獲取所有迭代的預測
            flow_predictions = self.model(
                img1_padded, 
                img2_padded, 
                img1_context_padded, 
                iters=self.max_iters, 
                test_mode=False  # 返回完整預測序列
            )
            
            # 計算每次迭代的色彩預測和loss
            iter_losses = []
            iter_colors = []
            
            for iter_idx, flow_pred in enumerate(flow_predictions):
                # 去除padding
                flow_pred = padder.unpad(flow_pred)
                
                # 色彩傳播
                colored_ab = self.warp_color_by_flow(img1_ab_tensor, flow_pred)
                
                # 計算L1 loss
                l1_loss = torch.abs(colored_ab - gt_ab_tensor).mean().item()
                iter_losses.append(l1_loss)
                iter_colors.append(colored_ab)
            
            # 計算最終預測的光流幅度（用於分組）
            final_flow = padder.unpad(flow_predictions[-1])
            flow_magnitude = torch.sqrt(final_flow[:, 0]**2 + final_flow[:, 1]**2).mean().item()
        
        # 構建結果
        result = {
            'scene_id': frame_pair['scene_id'],
            'scene': frame_pair['scene'],
            'dataset_idx': frame_pair['dataset_idx'],
            'frame_idx': frame_pair['frame_idx'],
            'frame_pair': f"{frame_pair['frame1_name']} -> {frame_pair['frame2_name']}",
            'iter_losses': iter_losses,  # list of floats
            'flow_magnitude': flow_magnitude,
            'final_loss': iter_losses[-1],
            'initial_loss': iter_losses[0],
            'improvement': iter_losses[0] - iter_losses[-1],
            'improvement_pct': (iter_losses[0] - iter_losses[-1]) / iter_losses[0] * 100 if iter_losses[0] > 0 else 0
        }
        
        return result
    
    def run_analysis(self, input_paths, num_samples=None, sample_per_scene=None):
        """
        運行完整分析
        
        Args:
            input_paths: list of str - 數據集路徑列表
            num_samples: int - 總採樣數量（None表示全部）
            sample_per_scene: int - 每個場景採樣數量
        """
        print("=" * 60)
        print("🎯 RAFT迭代收斂分析")
        print("=" * 60)
        print(f"📁 數據集數量: {len(input_paths)}")
        print(f"🔢 最大迭代次數: {self.max_iters}")
        print(f"📏 圖像尺寸: {self.image_size}")
        
        # 查找所有幀對
        frame_pairs = self.find_frame_pairs(input_paths)
        
        if len(frame_pairs) == 0:
            print("❌ 未找到任何幀對")
            return
        
        # 採樣策略
        if sample_per_scene is not None:
            # 按場景採樣
            sampled_pairs = []
            scene_groups = defaultdict(list)
            for pair in frame_pairs:
                scene_groups[pair['scene_id']].append(pair)
            
            for scene_id, pairs in scene_groups.items():
                n_sample = min(sample_per_scene, len(pairs))
                sampled_pairs.extend(np.random.choice(pairs, n_sample, replace=False))
            
            frame_pairs = sampled_pairs
            print(f"🎲 按場景採樣: 每場景{sample_per_scene}個，共{len(frame_pairs)}個")
        
        elif num_samples is not None and num_samples < len(frame_pairs):
            # 隨機採樣
            frame_pairs = np.random.choice(frame_pairs, num_samples, replace=False).tolist()
            print(f"🎲 隨機採樣: {len(frame_pairs)}個幀對")
        
        else:
            print(f"📊 使用全部數據: {len(frame_pairs)}個幀對")
        
        # 處理每個幀對
        print(f"\n🔄 開始分析 {len(frame_pairs)} 個幀對...")
        
        results = []
        failed_count = 0
        
        for pair in tqdm(frame_pairs, desc="分析進度"):
            result = self.analyze_frame_pair(pair)
            
            if result is not None:
                results.append(result)
                self.all_losses.append(result['iter_losses'])
                self.scene_losses[result['scene_id']].append(result['iter_losses'])
                
                # 按光流幅度分組
                if result['flow_magnitude'] < 5:
                    flow_group = 'low'
                elif result['flow_magnitude'] < 15:
                    flow_group = 'medium'
                else:
                    flow_group = 'high'
                self.flow_magnitude_losses[flow_group].append(result['iter_losses'])
            else:
                failed_count += 1
        
        print(f"\n✅ 分析完成!")
        print(f"   成功: {len(results)} 個")
        print(f"   失敗: {failed_count} 個")
        
        return results
    
    def compute_statistics(self):
        """計算統計數據"""
        print("\n" + "=" * 60)
        print("📊 計算統計數據...")
        print("=" * 60)
        
        if len(self.all_losses) == 0:
            print("❌ 沒有數據")
            return None
        
        all_losses_array = np.array(self.all_losses)  # [N, max_iters]
        
        stats = {
            'mean_losses': all_losses_array.mean(axis=0).tolist(),
            'median_losses': np.median(all_losses_array, axis=0).tolist(),
            'std_losses': all_losses_array.std(axis=0).tolist(),
            'percentile_25': np.percentile(all_losses_array, 25, axis=0).tolist(),
            'percentile_75': np.percentile(all_losses_array, 75, axis=0).tolist(),
            'min_losses': all_losses_array.min(axis=0).tolist(),
            'max_losses': all_losses_array.max(axis=0).tolist(),
            
            # 邊際收益
            'marginal_gains': [],
            'marginal_gains_pct': []
        }
        
        # 計算邊際收益
        mean_losses = stats['mean_losses']
        for i in range(1, len(mean_losses)):
            gain = mean_losses[i-1] - mean_losses[i]
            gain_pct = gain / mean_losses[i-1] * 100 if mean_losses[i-1] > 0 else 0
            stats['marginal_gains'].append(gain)
            stats['marginal_gains_pct'].append(gain_pct)
        
        # 按場景統計
        stats['by_scene'] = {}
        for scene_id, losses_list in self.scene_losses.items():
            scene_losses = np.array(losses_list)
            stats['by_scene'][scene_id] = {
                'mean_losses': scene_losses.mean(axis=0).tolist(),
                'count': len(losses_list)
            }
        
        # 按光流幅度統計
        stats['by_flow_magnitude'] = {}
        for flow_group, losses_list in self.flow_magnitude_losses.items():
            if len(losses_list) > 0:
                flow_losses = np.array(losses_list)
                stats['by_flow_magnitude'][flow_group] = {
                    'mean_losses': flow_losses.mean(axis=0).tolist(),
                    'count': len(losses_list)
                }
        
        # 打印關鍵統計
        print("\n📈 平均Loss變化:")
        for i, loss in enumerate(stats['mean_losses'], 1):
            if i == 1:
                print(f"   Iter {i:2d}: {loss:.6f}")
            else:
                gain = stats['marginal_gains'][i-2]
                gain_pct = stats['marginal_gains_pct'][i-2]
                print(f"   Iter {i:2d}: {loss:.6f} (↓{gain:.6f}, {gain_pct:.2f}%)")
        
        print(f"\n💡 總提升: {stats['mean_losses'][0]:.6f} -> {stats['mean_losses'][-1]:.6f}")
        print(f"   改善: {stats['mean_losses'][0] - stats['mean_losses'][-1]:.6f} ({(stats['mean_losses'][0] - stats['mean_losses'][-1])/stats['mean_losses'][0]*100:.2f}%)")
        
        # 建議的迭代次數
        threshold = 0.001  # 邊際收益閾值
        suggested_iters = self.max_iters
        for i, gain in enumerate(stats['marginal_gains']):
            if gain < threshold:
                suggested_iters = i + 1
                break
        
        print(f"\n🎯 建議迭代次數: {suggested_iters} (邊際收益閾值: {threshold:.4f})")
        
        return stats
    
    def visualize_results(self, stats, output_dir):
        """生成可視化圖表"""
        print(f"\n📊 生成可視化圖表...")
        os.makedirs(output_dir, exist_ok=True)
        
        iterations = list(range(1, self.max_iters + 1))
        
        # 圖1: Loss收斂曲線
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, stats['mean_losses'], 'b-o', linewidth=2, label='Mean', markersize=6)
        plt.fill_between(iterations, 
                        stats['percentile_25'], 
                        stats['percentile_75'], 
                        alpha=0.3, label='25-75 percentile')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('L1 Loss', fontsize=12)
        plt.title('Loss Convergence Across Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_convergence.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: loss_convergence.png")
        
        # 圖2: 邊際收益
        plt.figure(figsize=(12, 6))
        iter_gains = list(range(2, self.max_iters + 1))
        plt.bar(iter_gains, stats['marginal_gains'], color='steelblue', alpha=0.7)
        plt.axhline(y=0.001, color='r', linestyle='--', label='Threshold (0.001)')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Marginal Gain (Loss Reduction)', fontsize=12)
        plt.title('Marginal Gain per Iteration', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'marginal_gains.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: marginal_gains.png")
        
        # 圖3: 按光流幅度分組
        if len(stats['by_flow_magnitude']) > 0:
            plt.figure(figsize=(12, 6))
            for flow_group in ['low', 'medium', 'high']:
                if flow_group in stats['by_flow_magnitude']:
                    losses = stats['by_flow_magnitude'][flow_group]['mean_losses']
                    count = stats['by_flow_magnitude'][flow_group]['count']
                    plt.plot(iterations, losses, marker='o', linewidth=2, 
                            label=f'{flow_group.capitalize()} motion (n={count})')
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('L1 Loss', fontsize=12)
            plt.title('Loss Convergence by Motion Magnitude', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_by_motion.png'), dpi=300)
            plt.close()
            print(f"   ✓ 保存: loss_by_motion.png")
        
        # 圖4: 百分比改善
        plt.figure(figsize=(12, 6))
        iter_gains = list(range(2, self.max_iters + 1))
        plt.bar(iter_gains, stats['marginal_gains_pct'], color='coral', alpha=0.7)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.title('Percentage Improvement per Iteration', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_percentage.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: improvement_percentage.png")
        
        print(f"\n✅ 所有圖表已保存到: {output_dir}")
    
    def save_results(self, results, stats, output_dir):
        """保存詳細結果"""
        print(f"\n💾 保存結果...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存統計數據 (JSON)
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   ✓ 統計數據: {stats_path}")
        
        # 保存詳細結果 (CSV)
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, 'detailed_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"   ✓ 詳細結果: {results_path}")
        
        # 保存彙總報告 (TXT)
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAFT迭代收斂分析報告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"總樣本數: {len(results)}\n")
            f.write(f"最大迭代次數: {self.max_iters}\n")
            f.write(f"圖像尺寸: {self.image_size}\n\n")
            
            f.write("平均Loss變化:\n")
            f.write("-" * 60 + "\n")
            for i, loss in enumerate(stats['mean_losses'], 1):
                if i == 1:
                    f.write(f"Iter {i:2d}: {loss:.6f}\n")
                else:
                    gain = stats['marginal_gains'][i-2]
                    gain_pct = stats['marginal_gains_pct'][i-2]
                    f.write(f"Iter {i:2d}: {loss:.6f} (↓{gain:.6f}, {gain_pct:.2f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"初始Loss: {stats['mean_losses'][0]:.6f}\n")
            f.write(f"最終Loss: {stats['mean_losses'][-1]:.6f}\n")
            f.write(f"總改善: {stats['mean_losses'][0] - stats['mean_losses'][-1]:.6f} ")
            f.write(f"({(stats['mean_losses'][0] - stats['mean_losses'][-1])/stats['mean_losses'][0]*100:.2f}%)\n")
            
            # 建議
            threshold = 0.001
            suggested_iters = self.max_iters
            for i, gain in enumerate(stats['marginal_gains']):
                if gain < threshold:
                    suggested_iters = i + 1
                    break
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("建議與結論:\n")
            f.write("-" * 60 + "\n")
            f.write(f"建議迭代次數: {suggested_iters}\n")
            f.write(f"依據: 邊際收益低於閾值 {threshold:.4f}\n")
            
            if suggested_iters < self.max_iters:
                speedup = self.max_iters / suggested_iters
                f.write(f"預期速度提升: {speedup:.2f}x\n")
                quality_loss = stats['mean_losses'][suggested_iters-1] - stats['mean_losses'][-1]
                quality_loss_pct = quality_loss / stats['mean_losses'][-1] * 100
                f.write(f"質量損失: {quality_loss:.6f} ({quality_loss_pct:.2f}%)\n")
        
        print(f"   ✓ 分析報告: {report_path}")
        print(f"\n✅ 所有結果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='RAFT迭代收斂分析工具')
    
    # 必要參數
    parser.add_argument('--model', required=True, help='訓練好的模型路徑')
    parser.add_argument('--data_paths', required=True, help='數據集路徑（逗號分隔多個路徑）')
    parser.add_argument('--output_dir', required=True, help='輸出文件夾路徑')
    
    # 模型參數
    parser.add_argument('--small', action='store_true', help='使用small版本模型')
    parser.add_argument('--max_iters', type=int, default=12, help='最大迭代次數')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='處理圖像尺寸')
    
    # 採樣參數
    parser.add_argument('--num_samples', type=int, help='總採樣數量（不指定則使用全部）')
    parser.add_argument('--sample_per_scene', type=int, help='每個場景採樣數量')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 檢查文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    # 解析數據路徑
    data_paths = [path.strip() for path in args.data_paths.split(',')]
    
    # 檢查路徑
    valid_paths = []
    for path in data_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"⚠️ 路徑不存在，跳過: {path}")
    
    if len(valid_paths) == 0:
        print("❌ 沒有有效的數據路徑")
        return
    
    # 創建分析器
    analyzer = IterationAnalyzer(
        model_path=args.model,
        small_model=args.small,
        image_size=args.image_size,
        max_iters=args.max_iters
    )
    
    # 運行分析
    results = analyzer.run_analysis(
        input_paths=valid_paths,
        num_samples=args.num_samples,
        sample_per_scene=args.sample_per_scene
    )
    
    if results is None or len(results) == 0:
        print("❌ 分析失敗，沒有結果")
        return
    
    # 計算統計
    stats = analyzer.compute_statistics()
    
    if stats is None:
        print("❌ 統計計算失敗")
        return
    
    # 生成可視化
    analyzer.visualize_results(stats, args.output_dir)
    
    # 保存結果
    analyzer.save_results(results, stats, args.output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 示例用法
    if len(sys.argv) == 1:
        print("🎯 RAFT迭代收斂分析工具")
        print("\n📋 使用方法:")
        print("python iter_tool.py \\")
        print("    --model checkpoints/your-model.pth \\")
        print("    --data_paths /path/to/dataset1,/path/to/dataset2 \\")
        print("    --output_dir analysis_results \\")
        print("    --max_iters 12 \\")
        print("    --image_size 224 224")
        print("\n⚙️ 可選參數:")
        print("    --num_samples 500: 總採樣數量")
        print("    --sample_per_scene 10: 每個場景採樣數量")
        print("    --seed 42: 隨機種子")
        print("    --small: 使用small模型")
        print("\n📊 輸出內容:")
        print("    - loss_convergence.png: Loss收斂曲線")
        print("    - marginal_gains.png: 邊際收益圖")
        print("    - loss_by_motion.png: 按運動幅度分組")
        print("    - statistics.json: 詳細統計數據")
        print("    - detailed_results.csv: 每個樣本的結果")
        print("    - analysis_report.txt: 分析報告與建議")
    else:
        main()



