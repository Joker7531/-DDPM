"""
诊断推理流程问题

运行此脚本检查推理流程中的每个步骤
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')

from model import SpectrogramNAFNet
from inference import STFTInferenceProcessor

def diagnose_inference():
    """诊断推理流程"""
    print("=" * 60)
    print("推理流程诊断")
    print("=" * 60)
    
    # 1. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    model = SpectrogramNAFNet().to(device)
    model.eval()
    
    # 检查模型权重
    print("\n[1] 检查模型权重:")
    with torch.no_grad():
        # 检查输出层
        outro_weight = model.outro.weight
        outro_bias = model.outro.bias
        print(f"  输出层权重: mean={outro_weight.mean():.6f}, std={outro_weight.std():.6f}")
        print(f"  输出层偏置: mean={outro_bias.mean():.6f}")
        
        # 测试随机输入
        test_input = torch.randn(1, 2, 103, 156).to(device)
        test_output = model(test_input)
        print(f"  测试输出 (随机输入): mean={test_output.mean():.6f}, std={test_output.std():.6f}")
    
    # 2. 创建处理器
    processor = STFTInferenceProcessor(model, device)
    
    # 3. 创建模拟的真实STFT数据
    print("\n[2] 创建测试数据:")
    np.random.seed(42)
    
    # 模拟一个更真实的STFT数据 (EEG信号的STFT通常有特定的能量分布)
    # shape: [2, 257, 1000]
    t_frames = 1000
    raw_stft = np.zeros((2, 257, t_frames), dtype=np.float32)
    
    # 添加一些典型的EEG频率成分
    for freq_idx in range(1, 60):  # 主要能量在低频
        amplitude = 10.0 / (freq_idx + 1)  # 1/f 衰减
        phase = np.random.uniform(0, 2 * np.pi, t_frames)
        raw_stft[0, freq_idx, :] = amplitude * np.cos(phase)  # Real
        raw_stft[1, freq_idx, :] = amplitude * np.sin(phase)  # Imag
    
    # 添加一些噪声
    raw_stft += np.random.randn(2, 257, t_frames).astype(np.float32) * 0.01
    
    raw_magnitude = np.sqrt(raw_stft[0]**2 + raw_stft[1]**2)
    print(f"  原始STFT形状: {raw_stft.shape}")
    print(f"  原始幅度范围: [{raw_magnitude.min():.4f}, {raw_magnitude.max():.4f}]")
    print(f"  原始幅度均值: {raw_magnitude.mean():.4f}")
    
    # 4. 处理
    print("\n[3] 运行推理 (debug模式):")
    denoised_stft = processor.process_file(raw_stft, debug=True)
    
    # 5. 比较结果
    print("\n[4] 结果分析:")
    denoised_magnitude = np.sqrt(denoised_stft[0]**2 + denoised_stft[1]**2)
    print(f"  去噪幅度范围: [{denoised_magnitude.min():.4f}, {denoised_magnitude.max():.4f}]")
    print(f"  去噪幅度均值: {denoised_magnitude.mean():.4f}")
    
    ratio = denoised_magnitude / np.maximum(raw_magnitude, 1e-8)
    print(f"  幅度比例 (去噪/原始): mean={ratio.mean():.4f}, std={ratio.std():.4f}")
    
    # 6. ISTFT测试
    print("\n[5] ISTFT测试:")
    signal = processor.istft(denoised_stft)
    print(f"  重建信号长度: {len(signal)}")
    print(f"  重建信号范围: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"  重建信号RMS: {np.sqrt(np.mean(signal**2)):.4f}")
    
    # 检查是否接近零
    if np.abs(signal).max() < 1e-6:
        print("  ⚠️ 警告: 重建信号接近零!")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    diagnose_inference()
