#!/bin/bash
# 检查实际使用的 STFT 参数
# 运行这个脚本来验证参数

echo "检查 Dataset_STFT 目录中的数据..."
echo "====================================="

# 检查一个文件的形状
python3 << 'EOF'
import numpy as np
from pathlib import Path

stft_file = Path("Dataset_STFT/train/raw")
files = list(stft_file.glob("*.npy"))

if files:
    sample = np.load(files[0])
    print(f"样本文件: {files[0].name}")
    print(f"STFT 形状: {sample.shape}")
    
    # 从形状推断参数
    n_channels, n_freq, n_frames = sample.shape
    print(f"  通道数: {n_channels}")
    print(f"  频率bins: {n_freq}")
    print(f"  时间帧数: {n_frames}")
    
    # 推断 n_fft
    n_fft = (n_freq - 1) * 2
    print(f"\n推断的 STFT 参数:")
    print(f"  n_fft = {n_fft} (从频率bins推断)")
    print(f"  预期: n_fft = 512")
    
    if n_freq == 257:
        print("✓ 频率bins = 257, 符合 n_fft=512")
    else:
        print(f"⚠️  频率bins = {n_freq}, 不符合预期!")
else:
    print("未找到 STFT 文件")
EOF

echo ""
echo "建议:"
echo "1. 如果频率bins不是257,请重新生成STFT数据"
echo "2. 确认使用的命令: python precompute_stft.py --n_fft 512 --hop_length 64"
