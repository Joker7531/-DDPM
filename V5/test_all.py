"""
快速测试脚本 - 验证所有模块
无需真实数据集，使用随机数据
"""
import sys
from pathlib import Path
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("UAR-ACSSNet 模块测试")
print("="*70 + "\n")

# ==========================================
# 1. 测试 STFT 处理器
# ==========================================
print("1️⃣  测试 STFT 处理器...")
from signal_processing import STFTProcessor

stft_proc = STFTProcessor()
x_test = torch.randn(2, 1, 2048)
S_test = stft_proc(x_test)

print(f"   ✓ STFT: {x_test.shape} → {S_test.shape}")
print(f"   ✓ 频率 bins: {stft_proc.num_freq_bins}")
print(f"   ✓ 频率范围: [{stft_proc.k_min * stft_proc.df:.2f}, {stft_proc.k_max * stft_proc.df:.2f}] Hz\n")

# ==========================================
# 2. 测试数据集（使用临时目录）
# ==========================================
print("2️⃣  测试数据集模块...")
import tempfile
import numpy as np
from datasets import EEGPairDataset

# 创建临时数据集
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    
    # 创建目录结构
    for split in ["train", "val", "test"]:
        (tmpdir / split / "raw").mkdir(parents=True)
        (tmpdir / split / "clean").mkdir(parents=True)
        
        # 生成假数据
        for i in range(3):
            fname = f"{i:04d}.npy"
            np.save(tmpdir / split / "raw" / fname, np.random.randn(3000).astype(np.float32))
            np.save(tmpdir / split / "clean" / fname, np.random.randn(3000).astype(np.float32))
    
    # 测试数据集
    ds = EEGPairDataset(
        root=str(tmpdir),
        split="train",
        segment_length=2048,
        random_crop=True,
        normalize="zscore_per_sample",
        return_meta=True,
    )
    
    x_raw, x_clean, meta = ds[0]
    print(f"   ✓ Dataset length: {len(ds)}")
    print(f"   ✓ Sample shape: {x_raw.shape}, {x_clean.shape}")
    print(f"   ✓ Meta keys: {list(meta.keys())}\n")

# ==========================================
# 3. 测试模型
# ==========================================
print("3️⃣  测试 UAR-ACSSNet 模型...")
from models import UAR_ACSSNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

model = UAR_ACSSNet(
    segment_length=2048,
    unet_base_ch=32,
    unet_levels=4,
    spec_channels=64,
    acss_depth=3,
    num_freq_bins=103,
    dropout=0.1,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Total parameters: {total_params:,}")

# 前向传播
x_input = torch.randn(2, 1, 2048).to(device)

print(f"\n  Running forward pass (scan sanity checks will trigger)...")
with torch.no_grad():
    outputs = model(x_input)

print(f"\n   ✓ Forward pass completed")
print(f"   Output type: {type(outputs)}")
if isinstance(outputs, dict):
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Details:")
    for k, v in outputs.items():
        print(f"     - {k}: {v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
        if k == 'w':
            print(f"       w stats: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
else:
    print(f"   ⚠ Warning: Expected dict output, got {type(outputs)}")
print()

# ==========================================
# 4. 测试损失函数
# ==========================================
print("4️⃣  测试损失函数...")
from train import compute_losses

batch = (x_input, torch.randn(2, 1, 2048).to(device))
cfg = {
    "charbonnier_eps": 1e-6,
    "use_weighted_recon": False,
    "tv_weight": 0.01,
    "entropy_weight": 0.01,
    "recon_weight": 1.0,
    "conf_reg_weight": 0.1,
    "consistency_weight": 0.0,
}

losses = compute_losses(batch, outputs, cfg)

print(f"   ✓ Losses computed:")
for k, v in losses.items():
    print(f"     - {k}: {v.item():.6f}")
print()

# ==========================================
# 5. 测试训练循环（1 step）
# ==========================================
print("5️⃣  测试训练步骤...")
from torch.utils.data import TensorDataset, DataLoader

# 创建假数据加载器
dummy_ds = TensorDataset(
    torch.randn(16, 1, 2048),
    torch.randn(16, 1, 2048)
)
dummy_loader = DataLoader(dummy_ds, batch_size=4, shuffle=True)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练一步
model.train()
batch = next(iter(dummy_loader))
x_raw, x_clean = [b.to(device) for b in batch]

outputs = model(x_raw)
losses = compute_losses((x_raw, x_clean), outputs, cfg)
loss = losses["total"]

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"   ✓ Training step completed")
print(f"     - Loss: {loss.item():.6f}\n")

# ==========================================
# 6. 形状和范围检查
# ==========================================
print("6️⃣  形状和范围检查...")

# Shape assertions
assert outputs['y_hat'].shape == (x_raw.shape[0], 1, 2048), "y_hat shape mismatch"
assert outputs['w'].shape == (x_raw.shape[0], 1, 2048), "w shape mismatch"
assert outputs['g_freq'].shape[1] == 1, "g_freq channel mismatch"

# Range assertions
w_val = outputs['w']
assert (w_val >= 0).all() and (w_val <= 1.01).all(), "w out of range [0,1]"

print(f"   ✓ All shape assertions passed")
print(f"   ✓ All range assertions passed")

# 详细的 w 统计检查
print(f"\n   Confidence map (w) detailed check:")
print(f"     min:  {w_val.min():.6f}")
print(f"     max:  {w_val.max():.6f}")
print(f"     mean: {w_val.mean():.6f}")
print(f"     std:  {w_val.std():.6f}")

# 退化检查
warnings = []
if 0.49 < w_val.mean() < 0.51:
    warnings.append("w.mean() very close to 0.5")
if w_val.std() < 0.01:
    warnings.append("w.std() very small (< 0.01)")
if w_val.min() > 0.4 and w_val.max() < 0.6:
    warnings.append("w range very narrow [0.4, 0.6]")

if warnings:
    print(f"   ⚠ Potential issues detected:")
    for w in warnings:
        print(f"     - {w}")
else:
    print(f"   ✓ w statistics look healthy")
print()

# ==========================================
# 总结
# ==========================================
print("="*70)
print("✅ 所有模块测试通过！")
print("="*70)
print("\n📝 下一步:")
print("   1. 准备真实数据集（Dataset/train/val/test/raw/clean/*.npy）")
print("   2. 运行: python main.py --dataset_root <path> --quick_test")
print("   3. 完整训练: python main.py --dataset_root <path> --num_epochs 50")
print("\n" + "="*70 + "\n")
