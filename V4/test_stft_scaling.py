"""
测试 STFT/iSTFT 的缩放一致性

比较不同参数配置下的能量守恒情况
"""

import numpy as np
from scipy import signal

# 参数
fs = 500
n_fft = 512
hop_length = 64
nperseg = n_fft
noverlap = nperseg - hop_length

# 生成测试信号 (纯正弦波，便于验证)
duration = 2.0  # 秒
t = np.arange(0, duration, 1/fs)
test_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz 正弦波
original_energy = np.sum(test_signal ** 2)

print("=" * 70)
print("STFT/iSTFT 缩放测试")
print("=" * 70)
print(f"原始信号长度: {len(test_signal)}")
print(f"原始信号能量: {original_energy:.6f}")
print(f"原始信号幅度范围: [{test_signal.min():.6f}, {test_signal.max():.6f}]")
print()

# 测试1: 默认参数
print("测试 1: scipy.signal.stft 默认参数")
print("-" * 70)
f, t_stft, Zxx = signal.stft(
    test_signal,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    return_onesided=True
)
print(f"STFT 输出形状: {Zxx.shape}")
print(f"STFT 幅度范围: [{np.abs(Zxx).min():.6f}, {np.abs(Zxx).max():.6f}]")

# iSTFT 重建
t_recon, reconstructed = signal.istft(
    Zxx,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft
)
print(f"重建信号长度: {len(reconstructed)}")
print(f"重建信号幅度范围: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")

# 比较原始信号和重建信号 (取相同长度)
min_len = min(len(test_signal), len(reconstructed))
orig_trimmed = test_signal[:min_len]
recon_trimmed = reconstructed[:min_len]

reconstruction_error = np.mean((orig_trimmed - recon_trimmed) ** 2)
scaling_factor = np.mean(np.abs(orig_trimmed)) / np.mean(np.abs(recon_trimmed))

print(f"重建误差 (MSE): {reconstruction_error:.10f}")
print(f"缩放因子 (原始/重建): {scaling_factor:.6f}")
print()

# 测试2: 显式指定所有参数
print("测试 2: 显式指定 boundary='zeros', padded=True")
print("-" * 70)
f2, t2, Zxx2 = signal.stft(
    test_signal,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    boundary='zeros',
    padded=True,
    return_onesided=True
)

t2_recon, reconstructed2 = signal.istft(
    Zxx2,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    boundary=True,
    time_axis=-1,
    freq_axis=-2
)

min_len2 = min(len(test_signal), len(reconstructed2))
scaling_factor2 = np.mean(np.abs(test_signal[:min_len2])) / np.mean(np.abs(reconstructed2[:min_len2]))
print(f"缩放因子 (原始/重建): {scaling_factor2:.6f}")
print()

# 测试3: boundary=None (不填充边界)
print("测试 3: boundary=None (不填充边界)")
print("-" * 70)
f3, t3, Zxx3 = signal.stft(
    test_signal,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    boundary=None,
    padded=False,
    return_onesided=True
)

t3_recon, reconstructed3 = signal.istft(
    Zxx3,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=n_fft,
    boundary=False,
    time_axis=-1,
    freq_axis=-2
)

min_len3 = min(len(test_signal), len(reconstructed3))
scaling_factor3 = np.mean(np.abs(test_signal[:min_len3])) / np.mean(np.abs(reconstructed3[:min_len3]))
print(f"重建信号长度: {len(reconstructed3)}")
print(f"缩放因子 (原始/重建): {scaling_factor3:.6f}")
print()

# 检查 STFT 数据的实际缩放
print("STFT 数据分析:")
print("-" * 70)
stft_max = np.abs(Zxx).max()
signal_max = np.abs(test_signal).max()
print(f"时域信号最大值: {signal_max:.6f}")
print(f"STFT 幅度最大值: {stft_max:.6f}")
print(f"比值 (时域/频域): {signal_max / stft_max:.6f}")
print()

# 窗函数分析
window = signal.get_window('hann', nperseg)
window_sum = np.sum(window)
window_energy = np.sum(window ** 2)
print(f"Hann 窗参数:")
print(f"  窗长度: {len(window)}")
print(f"  窗和: {window_sum:.6f}")
print(f"  窗能量: {window_energy:.6f}")
print(f"  窗均方根: {np.sqrt(window_energy / len(window)):.6f}")
print()

print("=" * 70)
print("结论:")
print("=" * 70)
print(f"如果缩放因子接近 1.0: STFT/iSTFT 能量守恒良好")
print(f"如果缩放因子显著偏离 1.0: 存在缩放不一致问题")
print(f"当前观察到的缩放因子: {scaling_factor:.6f}")
if abs(scaling_factor - 1.0) > 0.01:
    print(f"⚠️  存在显著缩放问题！需要修正 iSTFT 的参数或手动缩放")
else:
    print(f"✓ 缩放基本一致")
