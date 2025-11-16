"""
loss.py - 視頻上色一致性損失函數
包含: L1 Sequence Loss, Perceptual Loss, Contextual Loss, Temporal Consistency Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============== 輔助函數 ==============

def feature_normalize(features):
    """L2 normalize features"""
    norm = torch.norm(features, p=2, dim=1, keepdim=True)
    return features / (norm + 1e-5)


def warp_color_by_flow(img_ab, flow):
    """
    使用光流進行色彩傳播
    
    Args:
        img_ab: [B, 2, H, W] - ab通道 [-1, 1]
        flow: [B, 2, H, W] - 光流
        
    Returns:
        warped_color: [B, 2, H, W] - 傳播後的色彩 [-1, 1]
    """
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
    
    # 歸一化到 [-1, 1]
    sampling_coords[..., 0] = 2.0 * sampling_coords[..., 0] / (W - 1) - 1.0
    sampling_coords[..., 1] = 2.0 * sampling_coords[..., 1] / (H - 1) - 1.0
    
    # 雙線性插值
    warped_color = F.grid_sample(
        img_ab, 
        sampling_coords, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return torch.clamp(warped_color, -1, 1)


def lab_to_rgb(lab):
    """
    簡化的LAB到RGB轉換
    
    Args:
        lab: [B, 3, H, W] - LAB格式
             L: [0, 100], a: [-128, 127], b: [-128, 127]
             
    Returns:
        rgb: [B, 3, H, W] - RGB格式 [0, 1]
    """
    # 如果有kornia,使用標準轉換
    try:
        import kornia
        return kornia.color.lab_to_rgb(lab)
    except ImportError:
        pass
    
    # 簡化版本 (近似)
    L = lab[:, 0:1, :, :] / 100.0  # [0, 1]
    a = lab[:, 1:2, :, :] / 127.0  # [-1, 1]
    b = lab[:, 2:3, :, :] / 127.0  # [-1, 1]
    
    # 簡單的線性近似
    r = L + 1.402 * b
    g = L - 0.344 * a - 0.714 * b
    b_channel = L + 1.772 * a
    
    rgb = torch.cat([r, g, b_channel], dim=1)
    return torch.clamp(rgb, 0, 1)


# ============== VGG特徵提取器 (修復版) ==============

class VGGFeatureExtractor(nn.Module):
    """
    VGG19特徵提取器 (修復版)
    使用slice分割,避免通道數錯誤
    """
    def __init__(self, layers=['relu2_2', 'relu3_4', 'relu4_4'], device='cuda'):
        super(VGGFeatureExtractor, self).__init__()
        
        # 載入預訓練的VGG19
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        # 凍結參數
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 🔥 關鍵修復: 分割成不同的slice
        self.slice1 = vgg[:4]    # input → relu1_2
        self.slice2 = vgg[4:9]   # relu1_2 → relu2_2
        self.slice3 = vgg[9:18]  # relu2_2 → relu3_4
        self.slice4 = vgg[18:27] # relu3_4 → relu4_4
        
        self.layers = layers
        
        # ImageNet標準化
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        print(f"✅ VGG19 Feature Extractor initialized")
        print(f"   Extracting layers: {layers}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] - RGB圖像 [0, 1]
            
        Returns:
            features: dict - {layer_name: feature_tensor}
        """
        # 確保輸入是3通道RGB
        if x.shape[1] != 3:
            raise ValueError(f"VGG expects 3-channel RGB input, got {x.shape[1]} channels")
        
        # ImageNet標準化
        x = (x - self.mean) / self.std
        
        features = {}
        
        # 🔥 逐層前向傳播 (每個slice接續上一個的輸出)
        h = self.slice1(x)
        if 'relu1_2' in self.layers:
            features['relu1_2'] = h
        
        h = self.slice2(h)
        if 'relu2_2' in self.layers:
            features['relu2_2'] = h
        
        h = self.slice3(h)
        if 'relu3_4' in self.layers:
            features['relu3_4'] = h
        
        h = self.slice4(h)
        if 'relu4_4' in self.layers:
            features['relu4_4'] = h
        
        return features


# ============== 1. L1 Sequence Loss ==============

def color_sequence_loss(color_preds, color_gt, gamma=0.8):
    """
    色彩序列損失函數 (RAFT專用)
    
    Args:
        color_preds: list of [B, 2, H, W] - 色彩預測序列 [-1, 1]
        color_gt: [B, 2, H, W] - 真實色彩 [-1, 1]
        gamma: 序列權重衰減因子
        
    Returns:
        loss: 總損失
        metrics: 評估指標
    """
    n_predictions = len(color_preds)
    color_loss = 0.0
    
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        color_diff = (color_preds[i] - color_gt).abs()
        i_loss = color_diff.mean()
        color_loss += i_weight * i_loss
    
    # 計算最終預測的評估指標
    final_pred = color_preds[-1]
    color_error = (final_pred - color_gt).abs().mean()
    
    error_per_pixel = torch.sqrt((final_pred - color_gt).pow(2).sum(dim=1))
    
    metrics = {
        'color_error': color_error.item(),
        '0.01_thresh': (error_per_pixel < 0.01).float().mean().item(),
        '0.02_thresh': (error_per_pixel < 0.02).float().mean().item(),
        '0.04_thresh': (error_per_pixel < 0.04).float().mean().item(),
    }
    
    return color_loss, metrics


# ============== 2. Perceptual Loss ==============

class PerceptualLoss(nn.Module):
    """
    感知損失 (Multi-layer VGG features)
    """
    def __init__(self, layers=['relu2_2', 'relu3_4', 'relu4_4'], 
                 layer_weights=None, device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        self.vgg = VGGFeatureExtractor(layers=layers, device=device)
        
        # 默認權重 (淺層權重小，深層權重大)
        if layer_weights is None:
            layer_weights = {
                'relu2_2': 0.1,
                'relu3_4': 0.2,
                'relu4_4': 0.3,
            }
        self.layer_weights = layer_weights
        
        print(f"✅ Perceptual Loss initialized")
        print(f"   Layer weights: {layer_weights}")
    
    def forward(self, pred_ab, gt_ab, L_channel):
        """
        Args:
            pred_ab: [B, 2, H, W] - 預測的ab通道 [-1, 1]
            gt_ab: [B, 2, H, W] - 真實的ab通道 [-1, 1]
            L_channel: [B, 1, H, W] - L通道 [0, 100]
            
        Returns:
            loss: perceptual loss
        """
        # 轉換ab範圍: [-1,1] → [-127,127]
        ab_pred = pred_ab * 127.0
        ab_gt = gt_ab * 127.0
        
        # 組合LAB
        lab_pred = torch.cat([L_channel, ab_pred], dim=1)
        lab_gt = torch.cat([L_channel, ab_gt], dim=1)
        
        # LAB → RGB
        rgb_pred = lab_to_rgb(lab_pred)
        rgb_gt = lab_to_rgb(lab_gt)
        
        # 提取VGG特徵
        feat_pred = self.vgg(rgb_pred)
        feat_gt = self.vgg(rgb_gt)
        
        # 計算多層loss
        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            if layer_name in feat_pred:
                layer_loss = F.mse_loss(feat_pred[layer_name], feat_gt[layer_name])
                loss += weight * layer_loss
        
        return loss


# ============== 3. Contextual Loss ==============

class ContextualLoss(nn.Module):
    """
    Contextual Loss (基於特徵相似度)
    用於保持紋理和結構一致性
    """
    def __init__(self, layer='relu3_4', h=0.1, device='cuda'):
        super(ContextualLoss, self).__init__()
        
        self.vgg = VGGFeatureExtractor(layers=[layer], device=device)
        self.layer = layer
        self.h = h  # bandwidth
        
        print(f"✅ Contextual Loss initialized")
        print(f"   Layer: {layer}, Bandwidth: {h}")
    
    def forward(self, pred_ab, gt_ab, L_channel, feature_centering=True):
        """
        Args:
            pred_ab: [B, 2, H, W] - 預測的ab通道 [-1, 1]
            gt_ab: [B, 2, H, W] - 真實的ab通道 [-1, 1]
            L_channel: [B, 1, H, W] - L通道 [0, 100]
            feature_centering: 是否中心化特徵
            
        Returns:
            loss: contextual loss
        """
        # 轉換ab範圍
        ab_pred = pred_ab * 127.0
        ab_gt = gt_ab * 127.0
        
        # 組合LAB
        lab_pred = torch.cat([L_channel, ab_pred], dim=1)
        lab_gt = torch.cat([L_channel, ab_gt], dim=1)
        
        # LAB → RGB
        rgb_pred = lab_to_rgb(lab_pred)
        rgb_gt = lab_to_rgb(lab_gt)
        
        # 提取VGG特徵
        feat_pred = self.vgg(rgb_pred)[self.layer]
        feat_gt = self.vgg(rgb_gt)[self.layer]
        
        return self._contextual_loss(feat_pred, feat_gt, self.h, feature_centering)
    
    def _contextual_loss(self, X_features, Y_features, h=0.1, feature_centering=True):
        """
        計算contextual loss
        """
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        
        # Feature centering
        if feature_centering:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        # Normalize features
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)
        
        # Cosine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)
        d = 1 - torch.matmul(X_features_permute, Y_features)
        
        # Normalized distance
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)
        
        # Pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        
        # Contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)
        return -torch.log(CX + 1e-5).mean()


# ============== 4. Temporal Consistency Loss ==============

class TemporalConsistencyLoss(nn.Module):
    """
    時序一致性損失 (不使用mask的簡化版本)
    """
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        print(f"✅ Temporal Consistency Loss initialized (No mask)")
    
    def forward(self, current_ab_pred, last_ab_pred, flow_forward):
        """
        Args:
            current_ab_pred: [B, 2, H, W] - 當前幀預測 [-1, 1]
            last_ab_pred: [B, 2, H, W] - 上一幀預測 [-1, 1]
            flow_forward: [B, 2, H, W] - 光流 (last -> current)
            
        Returns:
            loss: temporal consistency loss
        """
        # 用光流warp上一幀的預測到當前幀
        last_ab_warped = warp_color_by_flow(last_ab_pred, flow_forward)
        
        # 計算一致性loss (全圖)
        temporal_loss = F.mse_loss(current_ab_pred, last_ab_warped)
        
        return temporal_loss


# ============== 5. Composite Loss (總loss管理) ==============

class CompositeLoss(nn.Module):
    """
    組合損失函數 - 管理所有loss組件
    """
    def __init__(self, 
                 # L1 sequence參數
                 gamma=0.8,
                 # Perceptual參數
                 perceptual_layers=['relu2_2', 'relu3_4', 'relu4_4'],
                 perceptual_weights={'relu2_2': 0.1, 'relu3_4': 0.2, 'relu4_4': 0.3},
                 # Contextual參數
                 contextual_layer='relu3_4',
                 contextual_h=0.1,
                 # Loss權重
                 weight_l1=1.0,
                 weight_perceptual=0.0, #0.6
                 weight_contextual=0.0, #0.3
                 weight_temporal=0.0, #0.2
                 device='cuda'):
        
        super(CompositeLoss, self).__init__()
        
        self.gamma = gamma
        self.device = device
        
        # Loss權重
        self.weight_l1 = weight_l1
        self.weight_perceptual = weight_perceptual
        self.weight_contextual = weight_contextual
        self.weight_temporal = weight_temporal
        
        # 初始化各個loss模組
        self.perceptual_loss = PerceptualLoss(
            layers=perceptual_layers,
            layer_weights=perceptual_weights,
            device=device
        ) if weight_perceptual > 0 else None
        
        self.contextual_loss = ContextualLoss(
            layer=contextual_layer,
            h=contextual_h,
            device=device
        ) if weight_contextual > 0 else None
        
        self.temporal_loss = TemporalConsistencyLoss() if weight_temporal > 0 else None
        
        print("\n" + "="*60)
        print("🎯 Composite Loss initialized")
        print("="*60)
        print(f"Loss weights:")
        print(f"  - L1 Sequence:          {weight_l1}")
        print(f"  - Perceptual (VGG):     {weight_perceptual}")
        print(f"  - Contextual:           {weight_contextual}")
        print(f"  - Temporal Consistency: {weight_temporal}")
        print("="*60 + "\n")
    
    def forward(self, color_preds, color_gt, L_channel, 
                last_ab_pred=None, flow_forward=None):
        """
        計算總loss
        
        Args:
            color_preds: list of [B, 2, H, W] - RAFT的色彩預測序列
            color_gt: [B, 2, H, W] - 真實色彩
            L_channel: [B, 1, H, W] - L通道 [0, 100]
            last_ab_pred: [B, 2, H, W] - 上一幀預測 (用於temporal)
            flow_forward: [B, 2, H, W] - 光流 (用於temporal)
            
        Returns:
            total_loss: 總loss
            loss_dict: 各個loss的字典
        """
        losses = {}
        
        # 1. L1 Sequence Loss (必須)
        l1_loss, metrics = color_sequence_loss(color_preds, color_gt, self.gamma)
        losses['l1'] = l1_loss * self.weight_l1
        
        # 取最後一個預測用於其他loss
        final_pred = color_preds[-1]
        
        # 2. Perceptual Loss (可選)
        if self.perceptual_loss is not None and self.weight_perceptual > 0:
            perc_loss = self.perceptual_loss(final_pred, color_gt, L_channel)
            losses['perceptual'] = perc_loss * self.weight_perceptual
        
        # 3. Contextual Loss (可選)
        if self.contextual_loss is not None and self.weight_contextual > 0:
            ctx_loss = self.contextual_loss(final_pred, color_gt, L_channel)
            losses['contextual'] = ctx_loss * self.weight_contextual
        
        # 4. Temporal Consistency Loss (可選)
        if self.temporal_loss is not None and self.weight_temporal > 0:
            if last_ab_pred is not None and flow_forward is not None:
                temp_loss = self.temporal_loss(final_pred, last_ab_pred, flow_forward)
                losses['temporal'] = temp_loss * self.weight_temporal
        
        # 計算總loss
        total_loss = sum(losses.values())
        
        # 添加metrics到loss_dict
        for key, val in metrics.items():
            losses[key] = val
        
        return total_loss, losses
    
    def update_weights(self, **kwargs):
        """動態更新loss權重 (用於訓練策略調整)"""
        for key, value in kwargs.items():
            if hasattr(self, f'weight_{key}'):
                setattr(self, f'weight_{key}', value)
                print(f"✅ Updated weight_{key} = {value}")


# ============== 測試代碼 ==============

def test_losses():
    """測試各個loss的正確性"""
    print("="*60)
    print("Testing Loss Functions...")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 2, 256, 256
    
    # 模擬數據
    color_preds = [torch.randn(B, 2, H, W).to(device) * 0.5 for _ in range(12)]
    color_gt = torch.randn(B, 2, H, W).to(device) * 0.5
    L_channel = torch.rand(B, 1, H, W).to(device) * 100
    last_ab_pred = torch.randn(B, 2, H, W).to(device) * 0.5
    flow_forward = torch.randn(B, 2, H, W).to(device) * 10
    
    # 測試L1 loss
    print("\n1. Testing L1 Sequence Loss...")
    l1_loss, metrics = color_sequence_loss(color_preds, color_gt)
    print(f"   L1 Loss: {l1_loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    # 測試Perceptual loss
    print("\n2. Testing Perceptual Loss...")
    perc_loss_fn = PerceptualLoss(device=device)
    perc_loss = perc_loss_fn(color_preds[-1], color_gt, L_channel)
    print(f"   Perceptual Loss: {perc_loss.item():.4f}")
    
    # 測試Contextual loss
    print("\n3. Testing Contextual Loss...")
    ctx_loss_fn = ContextualLoss(device=device)
    ctx_loss = ctx_loss_fn(color_preds[-1], color_gt, L_channel)
    print(f"   Contextual Loss: {ctx_loss.item():.4f}")
    
    # 測試Temporal loss
    print("\n4. Testing Temporal Consistency Loss...")
    temp_loss_fn = TemporalConsistencyLoss()
    temp_loss = temp_loss_fn(color_preds[-1], last_ab_pred, flow_forward)
    print(f"   Temporal Loss: {temp_loss.item():.4f}")
    
    # 測試Composite loss
    print("\n5. Testing Composite Loss...")
    composite_loss_fn = CompositeLoss(device=device)
    total_loss, loss_dict = composite_loss_fn(
        color_preds, color_gt, L_channel,
        last_ab_pred, flow_forward
    )
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown:")
    for key, val in loss_dict.items():
        if isinstance(val, torch.Tensor):
            print(f"     - {key}: {val.item():.4f}")
        else:
            print(f"     - {key}: {val:.4f}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_losses()



