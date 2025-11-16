"""
SwinV2 Multi-Scale Projector (Modified)
將 SwinV2 的 4 個 Stage 特徵統一投影到相同維度 (128-dim)
Stage 4 特別處理: 7×7 降採樣到 5×5 以實現完整覆蓋
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageProjector(nn.Module):
    """
    單個 Stage 的投影頭
    將不同維度的特徵投影到統一的輸出維度
    """
    def __init__(self, in_dim, out_dim=128, hidden_dim=256):
        """
        Args:
            in_dim: 輸入特徵維度
                Stage 1: 96
                Stage 2: 192
                Stage 3: 384
                Stage 4: 768
            out_dim: 輸出特徵維度 (default: 128)
            hidden_dim: 中間層維度 (default: 256)
        """
        super(StageProjector, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 兩層 1x1 卷積 + ReLU
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_dim, H, W] 來自 SwinV2 某個 Stage 的特徵
        
        Returns:
            out: [B, out_dim, H, W] 投影後的特徵
        """
        return self.proj(x)


class Stage4Projector(nn.Module):
    """
    Stage 4 專用投影頭
    將 7×7 特徵降採樣到 5×5，同時投影維度
    """
    def __init__(self, in_dim=768, out_dim=128, hidden_dim=256):
        """
        Args:
            in_dim: 輸入特徵維度 (Stage 4: 768)
            out_dim: 輸出特徵維度 (default: 128)
            hidden_dim: 中間層維度 (default: 256)
        """
        super(Stage4Projector, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 方案: 使用 AdaptiveAvgPool2d 直接指定輸出尺寸
        # 這是最穩定的方式，保證從任意尺寸到 5×5
        self.proj = nn.Sequential(
            # 第一層: 維度投影
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            
            # 第二層: 空間降採樣 (7×7 → 5×5)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5)),  # 確保輸出 5×5
            
            # 第三層: 最終投影
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 768, 7, 7] 來自 SwinV2 Stage 4 的特徵
        
        Returns:
            out: [B, 128, 5, 5] 投影並降採樣後的特徵
        """
        return self.proj(x)


class MultiScaleProjector(nn.Module):
    """
    多尺度投影器
    管理 4 個 Stage 的投影頭，統一輸出維度
    Stage 4 特別處理: 7×7 → 5×5
    """
    def __init__(self, out_dim=128, hidden_dim=256):
        """
        Args:
            out_dim: 統一的輸出維度 (default: 128)
            hidden_dim: 投影頭的中間層維度 (default: 256)
        """
        super(MultiScaleProjector, self).__init__()
        
        self.out_dim = out_dim
        
        # SwinV2 各 Stage 的通道數 (SwinV2-Base 配置)
        stage_dims = {
            'stage1': 96,   # 1/4 解析度
            'stage2': 192,  # 1/8 解析度
            'stage3': 384,  # 1/16 解析度
            'stage4': 768   # 1/32 解析度
        }
        
        # 創建 4 個投影頭
        self.proj1 = StageProjector(stage_dims['stage1'], out_dim, hidden_dim)
        self.proj2 = StageProjector(stage_dims['stage2'], out_dim, hidden_dim)
        self.proj3 = StageProjector(stage_dims['stage3'], out_dim, hidden_dim)
        self.proj4 = Stage4Projector(stage_dims['stage4'], out_dim, hidden_dim)  # ⭐ 特殊處理
        
        print(f"✅ MultiScaleProjector initialized:")
        print(f"   Stage 1: {stage_dims['stage1']} → {out_dim} (spatial: keep)")
        print(f"   Stage 2: {stage_dims['stage2']} → {out_dim} (spatial: keep)")
        print(f"   Stage 3: {stage_dims['stage3']} → {out_dim} (spatial: keep)")
        print(f"   Stage 4: {stage_dims['stage4']} → {out_dim} (spatial: 7×7 → 5×5) ⭐")
    
    def forward(self, stage_features):
        """
        Args:
            stage_features: list of 4 tensors from SwinV2
                [0] stage1: [B, 96, H/4, W/4]
                [1] stage2: [B, 192, H/8, W/8]
                [2] stage3: [B, 384, H/16, W/16]
                [3] stage4: [B, 768, H/32, W/32] - 通常是 7×7
        
        Returns:
            projected_features: list of 4 tensors
                [0] proj_stage1: [B, 128, H/4, W/4]
                [1] proj_stage2: [B, 128, H/8, W/8]
                [2] proj_stage3: [B, 128, H/16, W/16]
                [3] proj_stage4: [B, 128, 5, 5] ⭐ 固定輸出 5×5
        """
        assert len(stage_features) == 4, f"Expected 4 stages, got {len(stage_features)}"
        
        # 分別投影 4 個 Stage
        proj_s1 = self.proj1(stage_features[0])
        proj_s2 = self.proj2(stage_features[1])
        proj_s3 = self.proj3(stage_features[2])
        proj_s4 = self.proj4(stage_features[3])  # ⭐ 會自動降採樣到 5×5
        
        return [proj_s1, proj_s2, proj_s3, proj_s4]
    
    def get_output_dims(self):
        """返回各 Stage 輸出的空間維度 (相對於輸入圖像)"""
        return {
            'stage1': (1/4, 1/4),     # H/4, W/4
            'stage2': (1/8, 1/8),     # H/8, W/8
            'stage3': (1/16, 1/16),   # H/16, W/16
            'stage4': 'fixed_5x5'     # ⭐ 固定 5×5
        }
    
    def get_search_ranges(self, radius=4, input_size=224):
        """
        計算各 Stage 的實際搜索範圍 (全圖像素)
        
        Args:
            radius: correlation 查找半徑 (default: 4)
            input_size: 輸入圖像尺寸 (default: 224)
        
        Returns:
            dict: 各 Stage 的搜索範圍
        """
        return {
            'stage1': radius * (input_size / 56),   # ≈ ±16 pixels
            'stage2': radius * (input_size / 28),   # ≈ ±32 pixels
            'stage3': radius * (input_size / 14),   # ≈ ±64 pixels
            'stage4': radius * (input_size / 5),    # ≈ ±179 pixels ⭐
        }


class LightweightProjector(nn.Module):
    """
    輕量級投影頭 (可選)
    只用單層 1x1 卷積，減少參數量
    """
    def __init__(self, in_dim, out_dim=128):
        super(LightweightProjector, self).__init__()
        
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
        
        # 初始化
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x):
        return self.proj(x)


class LightweightStage4Projector(nn.Module):
    """
    Stage 4 輕量級投影頭
    單層投影 + 降採樣到 5×5
    """
    def __init__(self, in_dim=768, out_dim=128):
        super(LightweightStage4Projector, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d((5, 5))  # 降採樣到 5×5
        )
        
        nn.init.kaiming_normal_(self.proj[0].weight, mode='fan_out', nonlinearity='relu')
        if self.proj[0].bias is not None:
            nn.init.constant_(self.proj[0].bias, 0)
    
    def forward(self, x):
        return self.proj(x)


class LightweightMultiScaleProjector(nn.Module):
    """
    輕量級多尺度投影器
    使用單層卷積代替兩層，減少約 50% 參數量
    Stage 4 降採樣到 5×5
    """
    def __init__(self, out_dim=128):
        super(LightweightMultiScaleProjector, self).__init__()
        
        self.out_dim = out_dim
        
        stage_dims = {
            'stage1': 96,
            'stage2': 192,
            'stage3': 384,
            'stage4': 768
        }
        
        self.proj1 = LightweightProjector(stage_dims['stage1'], out_dim)
        self.proj2 = LightweightProjector(stage_dims['stage2'], out_dim)
        self.proj3 = LightweightProjector(stage_dims['stage3'], out_dim)
        self.proj4 = LightweightStage4Projector(stage_dims['stage4'], out_dim)  # ⭐ 特殊處理
        
        print(f"✅ LightweightMultiScaleProjector initialized (reduced params)")
        print(f"   Stage 4: 7×7 → 5×5 downsampling enabled ⭐")
    
    def forward(self, stage_features):
        proj_s1 = self.proj1(stage_features[0])
        proj_s2 = self.proj2(stage_features[1])
        proj_s3 = self.proj3(stage_features[2])
        proj_s4 = self.proj4(stage_features[3])  # ⭐ 自動降採樣
        
        return [proj_s1, proj_s2, proj_s3, proj_s4]


# ===== 測試代碼 =====
def test_projector():
    """測試投影器的正確性"""
    print("="*60)
    print("Testing MultiScaleProjector with 5×5 Stage 4...")
    print("="*60)
    
    # 模擬 SwinV2 的 4 個 Stage 輸出
    B = 2
    H, W = 224, 224
    
    stage1 = torch.randn(B, 96, 56, 56)     # H/4, W/4
    stage2 = torch.randn(B, 192, 28, 28)    # H/8, W/8
    stage3 = torch.randn(B, 384, 14, 14)    # H/16, W/16
    stage4 = torch.randn(B, 768, 7, 7)      # H/32, W/32 ⭐
    
    stage_features = [stage1, stage2, stage3, stage4]
    
    print(f"\n📥 Input shapes:")
    for i, feat in enumerate(stage_features, 1):
        print(f"   Stage {i}: {list(feat.shape)}")
    
    # 測試標準投影器
    print("\n" + "-"*60)
    print("Testing Standard MultiScaleProjector...")
    print("-"*60)
    
    projector = MultiScaleProjector(out_dim=128, hidden_dim=256)
    projected_features = projector(stage_features)
    
    print(f"\n📤 Output shapes:")
    for i, feat in enumerate(projected_features, 1):
        print(f"   Stage {i}: {list(feat.shape)}")
    
    # 驗證維度正確性
    assert projected_features[0].shape == (B, 128, 56, 56), "Stage1 output shape mismatch"
    assert projected_features[1].shape == (B, 128, 28, 28), "Stage2 output shape mismatch"
    assert projected_features[2].shape == (B, 128, 14, 14), "Stage3 output shape mismatch"
    assert projected_features[3].shape == (B, 128, 5, 5), "Stage4 output shape mismatch (should be 5×5)" # ⭐
    
    print("\n✅ All dimension checks passed!")
    print(f"   ⭐ Stage 4 successfully downsampled: 7×7 → 5×5")
    
    # 計算參數量
    total_params = sum(p.numel() for p in projector.parameters())
    print(f"\n📊 Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 顯示搜索範圍
    search_ranges = projector.get_search_ranges(radius=4, input_size=224)
    print(f"\n🔍 Search ranges (radius=4, input=224):")
    for stage, range_px in search_ranges.items():
        if isinstance(range_px, str):
            print(f"   {stage}: {range_px}")
        else:
            print(f"   {stage}: ±{range_px:.1f} pixels")
    
    # 測試輕量級版本
    print("\n" + "="*60)
    print("Testing LightweightMultiScaleProjector...")
    print("="*60)
    
    lightweight_projector = LightweightMultiScaleProjector(out_dim=128)
    lightweight_features = lightweight_projector(stage_features)
    
    print(f"\n📤 Lightweight output shapes:")
    for i, feat in enumerate(lightweight_features, 1):
        print(f"   Stage {i}: {list(feat.shape)}")
    
    # 驗證 Stage 4
    assert lightweight_features[3].shape == (B, 128, 5, 5), "Lightweight Stage4 should be 5×5"
    
    lightweight_params = sum(p.numel() for p in lightweight_projector.parameters())
    print(f"\n📊 Lightweight parameters: {lightweight_params:,} ({lightweight_params/1e6:.2f}M)")
    print(f"   Reduction: {(1 - lightweight_params/total_params)*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_projector()



