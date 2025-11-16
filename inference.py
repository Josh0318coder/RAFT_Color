#!/usr/bin/env python3
"""
RAFT视频上色随机采样推理脚本 - 修正LAB色彩空间处理 + SwinV2输入标准化
- 随机采样指定数量的帧对
- 实时显示：当前彩色帧 | 推理下一帧 | 灰阶下一帧
- 自动保存到指定文件夹
- 修正了OpenCV LAB色彩空间的数值范围问题
- ✅ 修正了SwinV2输入的ImageNet标准化（匹配训练时的处理）
"""

import sys
sys.path.append('core')

import os
import glob
import random
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms  # ✅ 新增导入

from raft import RAFT
from utils.utils import InputPadder

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class VideoColorSampler:
    """视频上色采样器"""
    
    def __init__(self, model_path, small_model=False, image_size=[384, 512], iters=20):
        self.image_size = image_size
        self.iters = iters
        
        # ✅ 新增：ImageNet 标准化（与训练时一致）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 加载模型
        print(f"🔧 加载模型: {model_path}")
        self.model = self.load_model(model_path, small_model)
        print(f"✅ 模型已加载到 {DEVICE}")
    
    def load_model(self, model_path, small_model):
        """加载RAFT模型"""
        args = argparse.Namespace()
        args.small = small_model
        args.mixed_precision = False
        args.alternate_corr = False
        
        model = RAFT(args)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 处理checkpoint格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 移除'module.'前缀
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
    
    def find_frame_pairs(self, input_path):
        """查找所有可用的帧对"""
        print(f"🔍 扫描视频帧: {input_path}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        frame_pairs = []
        
        # 如果输入是文件夹，查找子文件夹中的视频序列
        if os.path.isdir(input_path):
            for scene_dir in os.listdir(input_path):
                scene_path = os.path.join(input_path, scene_dir)
                if os.path.isdir(scene_path):
                    # 获取该场景的所有帧
                    frames = []
                    for ext in image_extensions:
                        frames.extend(glob.glob(os.path.join(scene_path, f'*{ext}')))
                        frames.extend(glob.glob(os.path.join(scene_path, f'*{ext.upper()}')))
                    
                    frames = sorted(frames)
                    
                    # 生成连续帧对
                    for i in range(len(frames) - 1):
                        frame_pairs.append({
                            'scene': scene_dir,
                            'frame1_path': frames[i],
                            'frame2_path': frames[i + 1],
                            'frame1_name': os.path.basename(frames[i]),
                            'frame2_name': os.path.basename(frames[i + 1])
                        })
        
        print(f"📊 找到 {len(frame_pairs)} 个帧对")
        return frame_pairs
    
    def load_image_as_lab(self, image_path):
        """加载图像并转换为LAB格式 - 修正版本"""
        try:
            # 加载并调整尺寸
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            image_np = np.array(image, dtype=np.uint8)
            
            # 转换到LAB色彩空间
            image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            image_lab = image_lab.astype(np.float32)
            
            # OpenCV LAB格式转换为标准LAB范围
            image_lab[:, :, 0] = image_lab[:, :, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
            image_lab[:, :, 1] = image_lab[:, :, 1] - 128.0          # A: [0,255] -> [-128,127]
            image_lab[:, :, 2] = image_lab[:, :, 2] - 128.0          # B: [0,255] -> [-128,127]
            
            # 转为tensor [3, H, W]
            image_lab = torch.from_numpy(image_lab).permute(2, 0, 1)
            return image_lab
            
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            return None
    
    def prepare_raft_inputs(self, img_lab):
        """
        准备RAFT输入 - 修正版本（匹配训练时的处理）
        
        ✅ 关键修正：SwinV2输入需要ImageNet标准化
        """
        # 分离L和ab通道
        img_L = img_lab[0:1]  # [1, H, W] [0, 100]
        img_ab = img_lab[1:3]  # [2, H, W] [-128, 127]
        
        # ✅ 修正：L通道归一化到[0,1]，再做ImageNet标准化（与训练一致）
        img_L_norm = img_L / 100.0              # [0, 100] -> [0, 1]
        img_input = img_L_norm.repeat(3, 1, 1)  # [3, H, W] [0, 1]
        img_input = self.normalize(img_input)   # ✅ ImageNet标准化
        
        # Context输入: 完整LAB归一化到[-1, 1] 
        img_L_context = (img_L / 50.0) - 1.0    # L: [0,100] -> [-1,1] 
        img_ab_norm_context = img_ab / 127.0    # ab: [-128,127] -> [-1,1] 
        img_context_input = torch.cat([img_L_context, img_ab_norm_context], dim=0) # [3, H, W]
        
        # ab通道归一化到[-1, 1]
        img_ab_norm = img_ab / 127.0  # [-128,127] -> [-1, 1]
        
        return img_input, img_context_input, img_L, img_ab_norm
    
    def warp_color_by_flow(self, img1_ab, flow):
        """使用光流进行色彩传播"""
        B, _, H, W = flow.shape
        
        # 创建坐标网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=flow.dtype),
            torch.arange(W, device=flow.device, dtype=flow.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # 应用光流
        flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        sampling_coords = base_grid + flow_permuted
        
        # 归一化到[-1, 1]
        sampling_coords[..., 0] = 2.0 * sampling_coords[..., 0] / (W - 1) - 1.0
        sampling_coords[..., 1] = 2.0 * sampling_coords[..., 1] / (H - 1) - 1.0
        
        # 双线性插值采样
        warped_color = F.grid_sample(
            img1_ab, sampling_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return torch.clamp(warped_color, -1, 1)
    
    def lab_to_rgb(self, img_L, img_ab_norm):
        """LAB转RGB - 修正版本"""
        # 反归一化ab通道
        img_ab = img_ab_norm * 127.0  # [-1, 1] -> [-127, 127]
        
        # 组合LAB通道
        img_lab = torch.cat([img_L, img_ab], dim=0)  # [3, H, W]
        lab_np = img_lab.permute(1, 2, 0).numpy()
        
        # 转换回OpenCV LAB格式
        lab_cv = lab_np.copy()
        lab_cv[:, :, 0] = lab_np[:, :, 0] * 255.0 / 100.0  # L: [0,100] -> [0,255]
        lab_cv[:, :, 1] = lab_np[:, :, 1] + 128.0          # A: [-128,127] -> [0,255]
        lab_cv[:, :, 2] = lab_np[:, :, 2] + 128.0          # B: [-128,127] -> [0,255]
        
        # 确保范围正确并转为uint8
        lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)
        
        # 转换为RGB
        bgr_np = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        
        return rgb_np
    
    def test_color_roundtrip(self, image_path):
        """测试颜色处理的往返一致性"""
        print("=" * 50)
        print(f"测试图像: {image_path}")
        
        # 1. 加载原始RGB图像
        original_rgb = Image.open(image_path).convert('RGB')
        original_rgb = original_rgb.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
        original_np = np.array(original_rgb)
        
        # 2. 转换到LAB
        img_lab = self.load_image_as_lab(image_path)
        print(f"LAB ranges: L[{img_lab[0].min():.1f}, {img_lab[0].max():.1f}], "
              f"A[{img_lab[1].min():.1f}, {img_lab[1].max():.1f}], "
              f"B[{img_lab[2].min():.1f}, {img_lab[2].max():.1f}]")
        
        # 3. 经过RAFT输入准备
        img_input, _, img_L, img_ab_norm = self.prepare_raft_inputs(img_lab)
        print(f"SwinV2输入范围: [{img_input.min():.3f}, {img_input.max():.3f}]")
        print(f"归一化后AB范围: [{img_ab_norm.min():.3f}, {img_ab_norm.max():.3f}]")
        
        # 4. 直接转回RGB
        reconstructed_rgb = self.lab_to_rgb(img_L, img_ab_norm)
        
        # 5. 计算差异
        diff = np.abs(original_np.astype(float) - reconstructed_rgb.astype(float))
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"重建误差: 最大={max_diff:.2f}, 平均={mean_diff:.2f}")
        
        # 6. 显示对比
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(original_np)
        axes[0].set_title('原始RGB')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_rgb)
        axes[1].set_title('重建RGB')
        axes[1].axis('off')
        
        axes[2].imshow(diff.astype(np.uint8))
        axes[2].set_title(f'差异图 (max={max_diff:.1f})')
        axes[2].axis('off')
        
        # 显示L通道
        L_display = (img_L.squeeze().numpy() * 255 / 100).astype(np.uint8)
        axes[3].imshow(L_display, cmap='gray')
        axes[3].set_title('L通道')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if mean_diff > 15:
            print("⚠️ 警告: 重建误差较大")
        else:
            print("✅ 颜色往返测试通过")
            
        return mean_diff < 15
    
    def process_frame_pair(self, frame_pair):
        """处理单个帧对"""
        # 加载两帧
        img1_lab = self.load_image_as_lab(frame_pair['frame1_path'])
        img2_lab = self.load_image_as_lab(frame_pair['frame2_path'])
        
        if img1_lab is None or img2_lab is None:
            return None
        
        # 准备输入（✅ 已修正为使用ImageNet标准化）
        img1_input, img1_context_input, img1_L, img1_ab_norm = self.prepare_raft_inputs(img1_lab)
        img2_input, img2_context_input, img2_L, img2_ab_norm = self.prepare_raft_inputs(img2_lab)
        
        with torch.no_grad():
            # 转为GPU tensor
            img1_tensor = img1_input.unsqueeze(0).to(DEVICE)
            img2_tensor = img2_input.unsqueeze(0).to(DEVICE)
            img1_context_tensor = img1_context_input.unsqueeze(0).to(DEVICE)
            
            # Padding
            padder = InputPadder(img1_tensor.shape)
            img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)
            img1_context_padded = padder.pad(img1_context_tensor)[0]
            
            # RAFT推理
            flow_predictions = self.model(img1_padded, img2_padded, img1_context_padded, iters=self.iters, test_mode=True)
            
            if isinstance(flow_predictions, tuple):
                final_flow = flow_predictions[1]  # flow_up
            else:
                final_flow = flow_predictions[-1] if isinstance(flow_predictions, list) else flow_predictions
            
            # 去除padding
            final_flow = padder.unpad(final_flow)
            
            # 色彩传播
            img1_ab_tensor = img1_ab_norm.unsqueeze(0).to(DEVICE)
            colored_ab = self.warp_color_by_flow(img1_ab_tensor, final_flow)
            colored_ab_cpu = colored_ab.squeeze(0).cpu()
        
        # 生成三个版本的图像
        frame1_colored = self.lab_to_rgb(img1_L, img1_ab_norm)
        frame2_predicted = self.lab_to_rgb(img2_L, colored_ab_cpu)
        frame2_gray = self.lab_to_rgb(img2_L, torch.zeros_like(img2_ab_norm))
        
        return {
            'frame1_colored': frame1_colored,
            'frame2_predicted': frame2_predicted,
            'frame2_gray': frame2_gray,
            'frame_pair': frame_pair
        }
    
    def create_comparison_image(self, results):
        """创建三图对比"""
        frame1_colored = results['frame1_colored']
        frame2_predicted = results['frame2_predicted']
        frame2_gray = results['frame2_gray']
        
        # 创建水平拼接的对比图
        h, w = frame1_colored.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        comparison[:, :w] = frame1_colored
        comparison[:, w:2*w] = frame2_predicted
        comparison[:, 2*w:3*w] = frame2_gray
        
        return comparison
    
    def save_results(self, results, output_dir, sample_idx):
        """保存结果"""
        frame_pair = results['frame_pair']
        scene = frame_pair['scene']
        frame1_name = Path(frame_pair['frame1_name']).stem
        frame2_name = Path(frame_pair['frame2_name']).stem
        
        # 创建输出文件夹
        scene_dir = os.path.join(output_dir, 'samples')
        os.makedirs(scene_dir, exist_ok=True)
        
        # 文件名前缀
        prefix = f"sample_{sample_idx:04d}_{scene}_{frame1_name}_to_{frame2_name}"
        
        # 保存单独的图像
        Image.fromarray(results['frame1_colored']).save(
            os.path.join(scene_dir, f"{prefix}_01_current_colored.png"))
        Image.fromarray(results['frame2_predicted']).save(
            os.path.join(scene_dir, f"{prefix}_02_predicted_colored.png"))
        Image.fromarray(results['frame2_gray']).save(
            os.path.join(scene_dir, f"{prefix}_03_target_gray.png"))
        
        # 保存对比图
        comparison = self.create_comparison_image(results)
        Image.fromarray(comparison).save(
            os.path.join(scene_dir, f"{prefix}_comparison.png"))
        
        return os.path.join(scene_dir, f"{prefix}_comparison.png")
    
    def sample_and_inference(self, input_path, output_dir, num_samples, 
                           show_display=True, save_individual=True):
        """主要采样和推理函数"""
        print(f"🎬 开始随机采样推理")
        print(f"📁 输入路径: {input_path}")
        print(f"📁 输出路径: {output_dir}")
        print(f"🎯 采样数量: {num_samples}")
        print(f"🖥️  显示结果: {show_display}")
        
        # 查找所有帧对
        frame_pairs = self.find_frame_pairs(input_path)
        
        if len(frame_pairs) == 0:
            print("❌ 未找到任何帧对")
            return
        
        if num_samples > len(frame_pairs):
            print(f"⚠️  请求采样数({num_samples})大于可用帧对数({len(frame_pairs)})，使用全部帧对")
            num_samples = len(frame_pairs)
        
        # 随机采样
        sampled_pairs = random.sample(frame_pairs, num_samples)
        print(f"🎲 已随机选择 {len(sampled_pairs)} 个帧对")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        success_count = 0
        failed_count = 0
        
        # 处理每个采样的帧对
        for i, frame_pair in enumerate(tqdm(sampled_pairs, desc="处理帧对")):
            try:
                print(f"\n📸 处理样本 {i+1}/{num_samples}")
                print(f"   场景: {frame_pair['scene']}")
                print(f"   帧对: {frame_pair['frame1_name']} -> {frame_pair['frame2_name']}")
                
                # 推理
                results = self.process_frame_pair(frame_pair)
                
                if results is None:
                    print(f"⚠️  样本 {i+1} 处理失败")
                    failed_count += 1
                    continue
                
                # 保存结果
                if save_individual:
                    saved_path = self.save_results(results, output_dir, i+1)
                    print(f"💾 已保存: {os.path.basename(saved_path)}")
                
                # 显示结果
                if show_display:
                    self.display_results(results, sample_idx=i+1)
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 样本 {i+1} 处理出错: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        # 总结
        print(f"\n🎉 采样推理完成!")
        print(f"✅ 成功: {success_count} 个")
        print(f"❌ 失败: {failed_count} 个")
        print(f"📁 结果保存在: {output_dir}")
    
    def display_results(self, results, sample_idx):
        """显示结果对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 当前彩色帧
        axes[0].imshow(results['frame1_colored'])
        axes[0].set_title(f'当前彩色帧 (Frame {sample_idx})', fontsize=12)
        axes[0].axis('off')
        
        # 推理下一帧
        axes[1].imshow(results['frame2_predicted'])
        axes[1].set_title(f'推理下一帧 (Predicted)', fontsize=12, color='blue')
        axes[1].axis('off')
        
        # 灰阶下一帧
        axes[2].imshow(results['frame2_gray'])
        axes[2].set_title(f'灰阶下一帧 (Gray)', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f"样本 {sample_idx}: {results['frame_pair']['scene']}", fontsize=14)
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='RAFT视频上色随机采样推理 - 修正版（含SwinV2标准化）')
    
    # 必要参数
    parser.add_argument('--model', required=True, help='训练好的模型路径')
    parser.add_argument('--input_path', required=True, help='输入视频帧文件夹路径')
    parser.add_argument('--output_dir', required=True, help='输出文件夹路径')
    parser.add_argument('--num_samples', type=int, required=True, help='随机采样的帧对数量')
    
    # 模型参数
    parser.add_argument('--small', action='store_true', help='使用small版本模型')
    parser.add_argument('--iters', type=int, default=20, help='RAFT推理迭代次数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='处理图像尺寸')
    
    # 输出控制
    parser.add_argument('--no_display', action='store_true', help='不显示实时结果')
    parser.add_argument('--no_save', action='store_true', help='不保存单独图像文件')
    parser.add_argument('--seed', type=int, help='随机种子，用于复现采样结果')
    
    # 测试功能
    parser.add_argument('--test_color', action='store_true', help='仅运行颜色处理测试')
    parser.add_argument('--test_image', type=str, help='用于测试的图像路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 使用随机种子: {args.seed}")
    
    # 检查文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not args.test_color and not os.path.exists(args.input_path):
        print(f"❌ 输入路径不存在: {args.input_path}")
        return
    
    # 创建采样器
    sampler = VideoColorSampler(
        model_path=args.model,
        small_model=args.small,
        image_size=args.image_size,
        iters=args.iters
    )
    
    # 如果是测试模式
    if args.test_color:
        if not args.test_image or not os.path.exists(args.test_image):
            print("❌ 测试模式需要提供有效的测试图像路径 --test_image")
            return
        print("运行颜色处理测试...")
        sampler.test_color_roundtrip(args.test_image)
        return
    
    # 正常推理模式
    sampler.sample_and_inference(
        input_path=args.input_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        show_display=not args.no_display,
        save_individual=not args.no_save
    )


if __name__ == '__main__':
    # 示例用法
    if len(sys.argv) == 1:
        print("🎬 RAFT视频上色随机采样推理工具 - 修正版（含SwinV2标准化）")
        print("\n📋 使用方法:")
        print("python sample_inference_fixed.py \\")
        print("    --model checkpoints/your-model.pth \\")
        print("    --input_path /path/to/video/frames \\")
        print("    --output_dir results/samples \\")
        print("    --num_samples 50 \\")
        print("    --image_size 224 224")
        print("\n🧪 颜色测试模式:")
        print("python sample_inference_fixed.py \\")
        print("    --model checkpoints/your-model.pth \\")
        print("    --test_color \\")
        print("    --test_image /path/to/test/image.jpg \\")
        print("    --image_size 224 224")
        print("\n⚙️  可选参数:")
        print("    --small: 使用small模型")
        print("    --iters 20: 推理迭代次数")
        print("    --image_size 224 224: 处理尺寸（✅ 必须与训练时一致）")
        print("    --no_display: 不显示实时结果")
        print("    --no_save: 不保存单独文件")
        print("    --seed 42: 设置随机种子")
        print("\n🔧 修正内容:")
        print("    ✅ 正确处理OpenCV的LAB色彩空间格式")
        print("    ✅ 修正RGB->LAB和LAB->RGB的转换流程")
        print("    ✅ 添加SwinV2输入的ImageNet标准化（与训练一致）")
        print("    ✅ 添加颜色往返一致性测试功能")
    else:
        main()


