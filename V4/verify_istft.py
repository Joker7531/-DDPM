"""
验证实际 STFT 数据的 iSTFT 重建质量

加载一个实际的 raw STFT 文件，执行 iSTFT，
然后与原始时域信号对比，检查缩放是否正确
"""

import numpy as np
from scipy import signal
from pathlib import Path

# 参数 (与训练一致)
fs = 500
n_fft = 512
hop_length = 64
nperseg = n_fft
noverlap = nperseg - hop_length

# 路径 (根据实际情况调整)
stft_file = "Dataset_STFT/train/raw/train_092_raw.npy"
original_file = "Dataset/train/raw/train_092_raw.npy"  # 如果有的话

print("=" * 70)
print("实际 STFT 数据重建验证")
print("=" * 70)

# 加载 STFT 数据
stft_data = np.load(stft_file)
print(f"STFT 文件: {stft_file}")
print(f"STFT 形状: {stft_data.shape}")  # 应该是 [2, 257, T]
print(f"STFT 数据类型: {stft_data.dtype}")
print(f"STFT 幅度范围: [{np.abs(stft_data).min():.6f}, {np.abs(stft_data).max():.6f}]")
print()

# 转换为复数
real = stft_data[0]
imag = stft_data[1]
complex_stft = real + 1j * imag

print(f"复数 STFT 形状: {complex_stft.shape}")
print()

# 方法1: 不指定 boundary (默认行为)
print("方法 1: istft 默认参数")
print("-" * 70)
_, reconstructed1 = signal.istft(
    complex_stft,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft
)
print(f"重建信号长度: {len(reconstructed1)}")
print(f"重建信号幅度范围: [{reconstructed1.min():.6f}, {reconstructed1.max():.6f}]")
print(f"重建信号均值: {np.mean(reconstructed1):.6f}")
print(f"重建信号标准差: {np.std(reconstructed1):.6f}")
print()

# 方法2: boundary=True (匹配 stft 的 boundary='zeros')
print("方法 2: istft boundary=True")
print("-" * 70)
_, reconstructed2 = signal.istft(
    complex_stft,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    boundary=True,
    time_axis=-1,
    freq_axis=-2
)
print(f"重建信号长度: {len(reconstructed2)}")
print(f"重建信号幅度范围: [{reconstructed2.min():.6f}, {reconstructed2.max():.6f}]")
print(f"重建信号均值: {np.mean(reconstructed2):.6f}")
print(f"重建信号标准差: {np.std(reconstructed2):.6f}")
print()

# 比较两种方法
diff = np.mean(np.abs(reconstructed1 - reconstructed2[:len(reconstructed1)]))
print(f"两种方法的差异 (MAE): {diff:.6f}")
print()

# 如果有原始时域信号，加载并对比
if Path(original_file).exists():
    print("与原始时域信号对比:")
    print("-" * 70)
    original_signal = np.load(original_file)
    print(f"原始信号长度: {len(original_signal)}")
    print(f"原始信号幅度范围: [{original_signal.min():.6f}, {original_signal.max():.6f}]")
    print(f"原始信号标准差: {np.std(original_signal):.6f}")
    print()
    
    # 取相同长度
    min_len = min(len(original_signal), len(reconstructed2))
    orig_trimmed = original_signal[:min_len]
    recon_trimmed = reconstructed2[:min_len]
    
    # 计算缩放因子
    scaling_factor = np.std(orig_trimmed) / np.std(recon_trimmed)
    print(f"标准差比值 (原始/重建): {scaling_factor:.6f}")
    
    # MSE
    mse = np.mean((orig_trimmed - recon_trimmed) ** 2)
    print(f"重建误差 (MSE): {mse:.10f}")
    
    # 归一化后的相关系数
    corr = np.corrcoef(orig_trimmed, recon_trimmed)[0, 1]
    print(f"相关系数: {corr:.6f}")
    print()
    
    if abs(scaling_factor - 1.0) > 10:
        print(f"⚠️  警告: 缩放因子为 {scaling_factor:.2f}，存在显著缩放问题！")
        print(f"   需要对重建信号乘以缩放因子: {scaling_factor:.6f}")
    elif abs(scaling_factor - 1.0) > 1:
        print(f"⚠️  注意: 缩放因子为 {scaling_factor:.2f}，可能存在轻微缩放偏差")
    else:
        print(f"✓ 缩放因子接近 1.0，重建质量良好")
else:
    print(f"原始时域信号文件不存在: {original_file}")
    print("跳过与原始信号的对比")

print()
print("=" * 70)
print("提示:")
print("如果重建信号的幅度比原始信号小很多（例如 1/250），")
print("说明 scipy 的 STFT/iSTFT 能量归一化存在问题。")
print("解决方法:")
print("1. 确保 istft 的 boundary 参数正确")
print("2. 检查窗函数是否匹配")
print("3. 必要时手动添加缩放因子")
print("=" * 70)
