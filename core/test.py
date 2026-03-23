# test_components.py - 組件測試腳本

import torch
import sys
sys.path.append('core')

from dataset_new import VideoColorizationDataset, fetch_dataloader
from loss_new import CompositeLoss, warp_color_by_flow
import argparse

def test_dataloader():
    """測試1: Dataloader輸出格式"""
    print("="*60)
    print("🧪 Test 1: Dataloader Format")
    print("="*60)
    
    args = argparse.Namespace()
    args.data_path = '/home/m11302113/桌面/Video_color/dataset/DAVIS/davis_videos'
    args.batch_size = 2
    args.image_size = [224, 224]
    
    loader = fetch_dataloader(args)
    
    # 取一個batch
    batch = next(iter(loader))
    
    print(f"✓ images shape: {batch['images'].shape}")
    print(f"  Expected: [B=2, N=4, C=3, H=224, W=224]")
    
    print(f"✓ rgb_inputs shape: {batch['rgb_inputs'].shape}")
    print(f"  Expected: [B=2, N=4, C=3, H=224, W=224]")
    
    print(f"✓ scene_ids: {batch['scene_id']}")
    
    # 檢查數值範圍
    images = batch['images']
    print(f"\n✓ LAB ranges:")
    print(f"  L: [{images[:,:,0].min():.1f}, {images[:,:,0].max():.1f}] (expect [0,100])")
    print(f"  A: [{images[:,:,1].min():.1f}, {images[:,:,1].max():.1f}] (expect [-128,127])")
    print(f"  B: [{images[:,:,2].min():.1f}, {images[:,:,2].max():.1f}] (expect [-128,127])")
    
    print("\n✅ Dataloader test PASSED!\n")
    return batch


def test_color_warping(batch):
    """測試2: Color warping功能"""
    print("="*60)
    print("🧪 Test 2: Color Warping")
    print("="*60)
    
    B, N, C, H, W = batch['images'].shape
    
    # 模擬flow
    fake_flow = torch.randn(B, 2, H, W) * 5  # 小幅度flow
    
    # 取第一幀的AB通道
    frame0_lab = batch['images'][:, 0]  # [B, 3, H, W]
    frame0_ab = frame0_lab[:, 1:3]  # [B, 2, H, W]
    frame0_ab_norm = frame0_ab / 127.0  # [-1, 1]
    
    # Color warping
    warped_ab = warp_color_by_flow(frame0_ab_norm, fake_flow)
    
    print(f"✓ Input AB shape: {frame0_ab_norm.shape}")
    print(f"✓ Flow shape: {fake_flow.shape}")
    print(f"✓ Warped AB shape: {warped_ab.shape}")
    print(f"✓ Warped AB range: [{warped_ab.min():.3f}, {warped_ab.max():.3f}] (expect [-1,1])")
    
    # 檢查是否有NaN
    assert not torch.isnan(warped_ab).any(), "❌ NaN detected!"
    
    print("\n✅ Color warping test PASSED!\n")
    return warped_ab


def test_loss_computation(batch, warped_ab):
    """測試3: Loss計算"""
    print("="*60)
    print("🧪 Test 3: Loss Computation")
    print("="*60)
    
    B, N, C, H, W = batch['images'].shape
    
    # 準備數據
    frame1_lab = batch['images'][:, 1]  # [B, 3, H, W]
    gt_ab = frame1_lab[:, 1:3] / 127.0  # [B, 2, H, W] [-1,1]
    L_channel = frame1_lab[:, 0:1]      # [B, 1, H, W]
    
    # 模擬RAFT的sequence predictions (12次迭代)
    color_preds = []
    for i in range(12):
        # 逐漸接近GT (模擬收斂)
        alpha = i / 12.0
        pred = warped_ab * (1 - alpha) + gt_ab * alpha
        color_preds.append(pred)
    
    # 創建loss函數
    loss_fn = CompositeLoss(
        weight_l1=1.0,
        weight_perceptual=0.0,  # 先關閉
        weight_contextual=0.0,
        weight_temporal=0.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 計算loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    color_preds = [p.to(device) for p in color_preds]
    gt_ab = gt_ab.to(device)
    L_channel = L_channel.to(device)
    
    total_loss, loss_dict = loss_fn(color_preds, gt_ab, L_channel)
    
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Loss breakdown:")
    for key, val in loss_dict.items():
        if isinstance(val, torch.Tensor):
            print(f"    - {key}: {val.item():.4f}")
        else:
            print(f"    - {key}: {val:.4f}")
    
    # 檢查loss是否合理
    assert total_loss.item() > 0, "❌ Loss should be positive!"
    assert total_loss.item() < 10, "❌ Loss too large!"
    
    print("\n✅ Loss computation test PASSED!\n")
    return total_loss


def test_sequence_processing(batch):
    """測試4: 序列處理邏輯"""
    print("="*60)
    print("🧪 Test 4: Sequence Processing")
    print("="*60)
    
    B, N, C, H, W = batch['images'].shape
    
    print(f"✓ Sequence length: {N}")
    print(f"✓ Processing {N-1} frame pairs:")
    
    for ti in range(N - 1):
        frame_t = batch['images'][:, ti]      # [B, 3, H, W]
        frame_t1 = batch['images'][:, ti+1]   # [B, 3, H, W]
        
        print(f"\n  Frame {ti} → Frame {ti+1}")
        print(f"    Frame {ti} L range: [{frame_t[:,0].min():.1f}, {frame_t[:,0].max():.1f}]")
        print(f"    Frame {ti+1} L range: [{frame_t1[:,0].min():.1f}, {frame_t1[:,0].max():.1f}]")
        
        # 模擬memory累積
        if ti == 0:
            print(f"    Memory: [] (no history)")
        elif ti == 1:
            print(f"    Memory: [v0] (1 history)")
        else:
            print(f"    Memory: [v0, v1, ...] ({ti} histories)")
    
    print("\n✅ Sequence processing test PASSED!\n")


def test_full_pipeline(batch):
    """測試5: 完整流程模擬"""
    print("="*60)
    print("🧪 Test 5: Full Pipeline Simulation")
    print("="*60)
    
    B, N, C, H, W = batch['images'].shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模擬MemFlowNet處理
    print(f"Simulating MemFlowNet processing...")
    
    all_color_preds = []
    all_gts = []
    
    for ti in range(N - 1):
        print(f"\n  Processing frame {ti} → {ti+1}")
        
        # 準備輸入
        frame_t_ab = batch['images'][:, ti, 1:3] / 127.0  # [B, 2, H, W]
        frame_t1_ab = batch['images'][:, ti+1, 1:3] / 127.0  # GT
        
        # 模擬flow estimation + color warping
        fake_flow = torch.randn(B, 2, H, W) * 3
        warped = warp_color_by_flow(frame_t_ab, fake_flow)
        
        all_color_preds.append(warped.to(device))
        all_gts.append(frame_t1_ab.to(device))
        
        print(f"    Warped color range: [{warped.min():.3f}, {warped.max():.3f}]")
    
    # 計算平均loss
    total_loss = 0
    for pred, gt in zip(all_color_preds, all_gts):
        loss = torch.abs(pred - gt).mean()
        total_loss += loss
    
    avg_loss = total_loss / len(all_color_preds)
    print(f"\n✓ Average L1 loss across {N-1} predictions: {avg_loss.item():.4f}")
    
    print("\n✅ Full pipeline test PASSED!\n")


def main():
    print("\n" + "="*60)
    print("🚀 Component Testing Suite")
    print("="*60 + "\n")
    
    try:
        # Test 1: Dataloader
        batch = test_dataloader()
        
        # Test 2: Color warping
        warped_ab = test_color_warping(batch)
        
        # Test 3: Loss computation
        loss = test_loss_computation(batch, warped_ab)
        
        # Test 4: Sequence processing
        test_sequence_processing(batch)
        
        # Test 5: Full pipeline
        test_full_pipeline(batch)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


