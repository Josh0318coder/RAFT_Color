"""
raft.py - SwinV2 多尺度版本 (最小修改)
主要修改: 使用 4 個 SwinV2 stage 構建 4 個獨立的 CorrBlock
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder  # Context Encoder
from corr import CorrBlock, AlternateCorrBlock, MultiScaleCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

# SwinV2 相關
from timm import create_model
from swin_projector import MultiScaleProjector

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class SwinV2FeatureEncoder(nn.Module):
    """
    SwinV2-Tiny 特徵提取器
    - Backbone: 完全凍結，保持 eval 模式
    - Projector: 可訓練
    """
    def __init__(self, pretrained=True):
        super(SwinV2FeatureEncoder, self).__init__()
        
        # 載入 SwinV2-Tiny (224×224 預訓練)
        self.backbone = create_model(
            "swinv2_cr_tiny_ns_224.sw_in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=[-4, -3, -2, -1],  # 4個stage
        )
        
        # 凍結 backbone 參數
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 可訓練的投影器：4個stage → 128-dim
        self.projector = MultiScaleProjector(out_dim=128, hidden_dim=256)
        
        print("✅ SwinV2FeatureEncoder initialized")
        print("   - Backbone: SwinV2-Tiny (frozen, pretrained)")
        print("   - Projector: 4 stages → 128-dim (trainable)")
    
    def train(self, mode=True):
        """
        ✅ 關鍵修改：重寫 train() 方法
        確保 backbone 永遠保持 eval 模式
        """
        super().train(mode)
        
        # 強制 backbone 保持 eval 模式（凍結 Dropout, LayerNorm 等）
        self.backbone.eval()
        
        # Projector 跟隨訓練/評估模式
        self.projector.train(mode)
        
        return self
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: [B, 3, H, W] - ImageNet 標準化的灰階 RGB
        
        Returns:
            projected_features: list of 4 tensors
                [0]: [B, 128, H/4, W/4]   - Stage 1
                [1]: [B, 128, H/8, W/8]   - Stage 2
                [2]: [B, 128, H/16, W/16] - Stage 3
                [3]: [B, 128, 5, 5]       - Stage 4 (固定5×5)
        """
        # Backbone 前向（不需要 torch.no_grad，requires_grad=False 已生效）
        stage_features = self.backbone(x)
        
        # 投影到統一維度
        projected_features = self.projector(stage_features)
        
        return projected_features


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        # 設定維度
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # ===== Feature Encoder: 使用 SwinV2 =====
        print("🔧 Initializing Feature Encoder (SwinV2 Multi-Scale)...")
        self.fnet = SwinV2FeatureEncoder(pretrained=True)
        
        # ===== Context Encoder: 保持原始設計 =====
        print("🔧 Initializing Context Encoder (BasicEncoder)...")
        if args.small:
            self.cnet = SmallEncoder(
                output_dim=hdim+cdim, 
                norm_fn='none', 
                dropout=args.dropout
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.cnet = BasicEncoder(
                output_dim=hdim+cdim, 
                norm_fn='batch', 
                dropout=args.dropout
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        print(f"✅ RAFT initialized (Multi-Scale SwinV2 mode)")
        print(f"   hidden_dim={hdim}, context_dim={cdim}")
        print(f"   corr_levels=4 (4 SwinV2 stages), radius={args.corr_radius}")

    def freeze_bn(self):
        """凍結 BatchNorm（用於 Context Encoder）"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """初始化光流場為零（基於 Stage 2 解析度，即 1/8）"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """使用 convex upsampling 上採樣光流到原始解析度"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1_rgb, image2_rgb, context1_lab, 
                iters=12, flow_init=None, test_mode=False):
        """
        RAFT 前向傳播 (Multi-Scale SwinV2 版本)
        
        Args:
            image1_rgb: [B, 3, H, W] - ImageNet 標準化的灰階 RGB（給 SwinV2）
            image2_rgb: [B, 3, H, W] - ImageNet 標準化的灰階 RGB（給 SwinV2）
            context1_lab: [B, 3, H, W] - LAB 歸一化的彩色圖（給 Context Encoder）
            iters: 迭代次數
            flow_init: 初始光流（可選）
            test_mode: 測試模式
        
        Returns:
            flow_predictions: list of [B, 2, H, W] 光流預測序列
            或 (flow_low, flow_up) if test_mode=True
        """
        
        # 確保輸入連續
        image1_rgb = image1_rgb.contiguous()
        image2_rgb = image2_rgb.contiguous()
        context1_lab = context1_lab.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # ===== 1. Feature Extraction (SwinV2 Multi-Scale) =====
        with autocast(enabled=self.args.mixed_precision):
            fmap1_stages = self.fnet(image1_rgb)  # list of 4 stages
            fmap2_stages = self.fnet(image2_rgb)  # list of 4 stages
            
            # 轉為 float (如果使用混合精度)
            fmap1_stages = [f.float() for f in fmap1_stages]
            fmap2_stages = [f.float() for f in fmap2_stages]
        
        # ===== 2. Build Multi-Scale Correlation Volume =====
        # ✨ 關鍵修改：使用 MultiScaleCorrBlock 替代原始 CorrBlock
        corr_fn = MultiScaleCorrBlock(
            fmap1_stages, 
            fmap2_stages, 
            num_levels=self.args.corr_levels,  # 4
            radius=self.args.corr_radius
        )

        # ===== 3. Context Encoding =====
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(context1_lab)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # ===== 4. Initialize Flow =====
        # 注意：基於 Stage 2 解析度 (1/8)
        coords0, coords1 = self.initialize_flow(image1_rgb)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # ===== 5. Iterative Updates =====
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            
            # ✨ 查找 correlation (自動處理多尺度)
            corr = corr_fn(coords1)  # [B, 4*(2r+1)², H/8, W/8]

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # update_block 接收的輸入格式沒變
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # 更新坐標
            coords1 = coords1 + delta_flow

            # 上採樣到原始解析度
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        
        return flow_predictions



