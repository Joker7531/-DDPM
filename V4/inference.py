"""
SpectrogramNAFNet 推理脚本

本模块实现文件级推理，对输入的STFT频谱进行去噪，
并将结果反变换回一维时域信号。

支持功能:
- 单文件推理
- 批量文件推理
- ISTFT 反变换

作者: AI Assistant
日期: 2025-12-30
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SpectrogramNAFNet


class STFTInferenceProcessor:
    """
    STFT域推理处理器
    
    对完整STFT文件进行滑动窗口推理，并实现反归一化和ISTFT。
    使用50%重叠 + Hann窗加权平均，减少窗口边界伪影。
    
    Args:
        model: 训练好的SpectrogramNAFNet模型
        device: 计算设备
        window_size: 推理窗口大小 (帧数)
        overlap_ratio: 窗口重叠比例 (0.5 表示 50% 重叠)
        freq_start: 频率起始索引
        freq_end: 频率结束索引
        n_fft: STFT的FFT点数
        hop_length: STFT的跳跃长度
        sample_rate: 采样率
        eps: 数值稳定性常数
    """
    
    def __init__(
        self,
        model: SpectrogramNAFNet,
        device: torch.device,
        window_size: int = 156,
        overlap_ratio: float = 0.5,
        freq_start: int = 1,
        freq_end: int = 104,
        n_fft: int = 512,
        hop_length: int = 64,
        sample_rate: int = 500,
        eps: float = 1e-6
    ) -> None:
        self.model = model
        self.device = device
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(window_size * (1 - overlap_ratio))  # 50% overlap -> stride = 78
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.eps = eps
        
        # 预计算窗函数权重 (Hann窗用于平滑过渡)
        self.window_weights = self._create_window_weights()
        
        # 确保模型在评估模式
        self.model.eval()
    
    def _create_window_weights(self) -> np.ndarray:
        """
        创建用于重叠加权平均的窗函数
        
        使用Hann窗实现平滑过渡，减少窗口边界伪影
        
        Returns:
            窗函数权重，形状 [window_size]
        """
        # Hann窗: 0.5 * (1 - cos(2*pi*n/(N-1)))
        window = np.hanning(self.window_size).astype(np.float32)
        return window
    
    def _instance_normalize(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Instance Normalization: 对单个切片直接进行 Z-score 归一化
        
        直接对 Real/Imag 通道进行操作，不做幅度-相位分离，不做 Log 变换。
        与训练时的归一化策略保持一致。
        
        Args:
            data: 形状 [2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            (normalized_data, mean, std)
        """
        # 计算整个切片的统计量 (跨 Real/Imag 和所有频率、时间)
        mean = float(np.mean(data))
        std = float(np.std(data))
        std = max(std, self.eps)  # 防止除零
        
        # Z-score 归一化
        normalized_data = (data - mean) / std
        
        # 限制范围防止极端值
        normalized_data = np.clip(normalized_data, -10.0, 10.0)
        
        # 处理 NaN/Inf
        if not np.isfinite(normalized_data).all():
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized_data, mean, std
    
    def _instance_denormalize(
        self,
        normalized_data: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """
        反归一化：从归一化域恢复到原始STFT域
        
        简单的 Z-score 反变换：data = normalized_data * std + mean
        
        Args:
            normalized_data: 归一化域的数据 [2, F, T]
            mean: 归一化时的均值
            std: 归一化时的标准差
            
        Returns:
            反归一化后的STFT切片 [2, F, T]
        """
        # 反 Z-score
        denormalized = normalized_data * std + mean
        
        return denormalized
    
    def _pad_frequency(self, data: np.ndarray) -> np.ndarray:
        """
        将裁剪后的频率维度填充回原始大小
        
        Args:
            data: 形状 [2, 103, T] (裁剪后的频率维度)
            
        Returns:
            填充后的数据 [2, 257, T]
        """
        _, f_crop, t = data.shape
        full_freq = self.n_fft // 2 + 1  # 257
        
        result = np.zeros((2, full_freq, t), dtype=data.dtype)
        result[:, self.freq_start:self.freq_end, :] = data
        
        return result
    
    @torch.no_grad()
    def process_file(
        self,
        raw_stft: np.ndarray,
        debug: bool = False
    ) -> np.ndarray:
        """
        处理单个STFT文件 (50%重叠 + 加权平均)
        
        使用Hann窗进行加权平均，减少窗口边界伪影。
        
        Args:
            raw_stft: 原始STFT数据，形状 [2, 257, T_long]
            debug: 是否打印调试信息
            
        Returns:
            去噪后的STFT数据，形状 [2, 257, T_long]
        """
        _, full_freq, t_length = raw_stft.shape
        
        if debug:
            raw_mag = np.sqrt(raw_stft[0]**2 + raw_stft[1]**2)
            print(f"[DEBUG] 输入STFT: shape={raw_stft.shape}, "
                  f"幅度范围=[{raw_mag.min():.4f}, {raw_mag.max():.4f}]")
        
        # 计算切片数量
        if t_length < self.window_size:
            print(f"警告: 时间维度 {t_length} 小于窗口大小 {self.window_size}")
            return raw_stft.copy()
        
        # 加权累加缓冲区
        accum_buffer = np.zeros((2, full_freq, t_length), dtype=np.float32)
        weight_buffer = np.zeros(t_length, dtype=np.float32)
        
        # 计算完整窗口数量
        num_slices = (t_length - self.window_size) // self.stride + 1
        
        # 窗函数权重 (Hann窗)
        window_weights = self.window_weights  # [window_size]
        
        for i in tqdm(range(num_slices), desc="推理中", leave=False):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # 1. 提取切片并裁剪频率
            raw_slice_full = raw_stft[:, :, start_idx:end_idx]  # [2, 257, window]
            raw_slice = raw_slice_full[:, self.freq_start:self.freq_end, :]  # [2, 103, window]
            
            # 2. Instance Normalization (直接对 Real/Imag 做 Z-score)
            raw_norm, mean, std = self._instance_normalize(raw_slice)
            
            # 调试：第一个切片打印详细信息
            if debug and i == 0:
                print(f"[DEBUG] 切片0 归一化: mean={mean:.4f}, std={std:.4f}")
                print(f"[DEBUG] 切片0 raw_norm范围: [{raw_norm.min():.4f}, {raw_norm.max():.4f}]")
            
            # 3. 转换为tensor并推理 (模型预测噪声残差)
            raw_tensor = torch.from_numpy(raw_norm).float().unsqueeze(0).to(self.device)
            noise_pred = self.model(raw_tensor)
            noise_pred = noise_pred.squeeze(0).cpu().numpy()
            
            # 调试
            if debug and i == 0:
                print(f"[DEBUG] 切片0 模型预测噪声范围: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]")
            
            # 4. 计算去噪结果 (归一化域): denoised = raw - noise
            denoised_norm = raw_norm - noise_pred
            
            if debug and i == 0:
                print(f"[DEBUG] 切片0 去噪后归一化范围: [{denoised_norm.min():.4f}, {denoised_norm.max():.4f}]")
            
            # 5. 反归一化
            denoised_slice = self._instance_denormalize(denoised_norm, mean, std)
            
            if debug and i == 0:
                print(f"[DEBUG] 切片0 反归一化后范围: [{denoised_slice.min():.4f}, {denoised_slice.max():.4f}]")
                print(f"[DEBUG] 切片0 原始范围: [{raw_slice.min():.4f}, {raw_slice.max():.4f}]")
            
            # 6. 填充回原始频率维度
            denoised_slice_full = self._pad_frequency(denoised_slice)
            
            # 保留未处理频段的原始值
            denoised_slice_full[:, :self.freq_start, :] = raw_slice_full[:, :self.freq_start, :]
            denoised_slice_full[:, self.freq_end:, :] = raw_slice_full[:, self.freq_end:, :]
            
            # 7. 加权累加 (使用Hann窗权重)
            # 权重广播到所有频率
            accum_buffer[:, :, start_idx:end_idx] += denoised_slice_full * window_weights[np.newaxis, np.newaxis, :]
            weight_buffer[start_idx:end_idx] += window_weights
        
        # 处理末尾未覆盖的部分 (如果有)
        last_covered = (num_slices - 1) * self.stride + self.window_size
        if last_covered < t_length:
            # 处理最后一个窗口，从末尾往前取
            start_idx = t_length - self.window_size
            end_idx = t_length
            
            raw_slice_full = raw_stft[:, :, start_idx:end_idx]
            raw_slice = raw_slice_full[:, self.freq_start:self.freq_end, :]
            
            raw_norm, mean, std = self._instance_normalize(raw_slice)
            raw_tensor = torch.from_numpy(raw_norm).float().unsqueeze(0).to(self.device)
            noise_pred = self.model(raw_tensor)
            noise_pred = noise_pred.squeeze(0).cpu().numpy()
            
            # 计算去噪结果: denoised = raw - noise
            denoised_norm = raw_norm - noise_pred
            denoised_slice = self._instance_denormalize(denoised_norm, mean, std)
            denoised_slice_full = self._pad_frequency(denoised_slice)
            denoised_slice_full[:, :self.freq_start, :] = raw_slice_full[:, :self.freq_start, :]
            denoised_slice_full[:, self.freq_end:, :] = raw_slice_full[:, self.freq_end:, :]
            
            accum_buffer[:, :, start_idx:end_idx] += denoised_slice_full * window_weights[np.newaxis, np.newaxis, :]
            weight_buffer[start_idx:end_idx] += window_weights
        
        # 归一化：除以权重和
        # 对于权重为零的区域，使用原始数据
        weight_mask = weight_buffer > self.eps
        weight_buffer_safe = np.where(weight_mask, weight_buffer, 1.0)
        
        denoised_stft = accum_buffer / weight_buffer_safe[np.newaxis, np.newaxis, :]
        
        # 对于未覆盖的区域，使用原始数据
        for i in range(t_length):
            if not weight_mask[i]:
                denoised_stft[:, :, i] = raw_stft[:, :, i]
        
        if debug:
            denoised_mag = np.sqrt(denoised_stft[0]**2 + denoised_stft[1]**2)
            print(f"[DEBUG] 输出STFT幅度范围: [{denoised_mag.min():.4f}, {denoised_mag.max():.4f}]")
            print(f"[DEBUG] 权重缓冲区范围: [{weight_buffer.min():.4f}, {weight_buffer.max():.4f}]")
            uncovered = (~weight_mask).sum()
            if uncovered > 0:
                print(f"[DEBUG] 未覆盖帧数: {uncovered} (使用原始数据填充)")
        
        return denoised_stft
    
    def istft(
        self,
        stft_data: np.ndarray,
        window: str = 'hann'
    ) -> np.ndarray:
        """
        执行逆短时傅里叶变换 (ISTFT)
        
        使用与 scipy.signal.stft 兼容的方式进行重建。
        scipy.signal.istft 会自动处理窗函数的能量归一化。
        
        Args:
            stft_data: STFT数据，形状 [2, n_freq, n_frames]
                       Channel 0=Real, 1=Imag
            window: 窗函数类型
            
        Returns:
            重建的一维时域信号
        """
        # 方案：直接使用 scipy.signal.istft 以匹配数据准备时的 STFT
        from scipy import signal as scipy_signal
        
        # 转换为复数形式
        real = stft_data[0]  # [n_freq, n_frames]
        imag = stft_data[1]
        complex_stft = real + 1j * imag  # [n_freq, n_frames]
        
        # 使用 scipy.signal.istft（与数据准备时的 stft 对应）
        # 重要: boundary 参数必须与 stft 时保持一致
        # stft 默认 boundary='zeros' -> istft 需要 boundary=True
        _, reconstructed = scipy_signal.istft(
            complex_stft,
            fs=self.sample_rate,
            window=window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=True,  # 对应 stft 的 boundary='zeros'
            time_axis=-1,
            freq_axis=-2
        )
        
        return reconstructed.astype(np.float32)
    
    def process_and_reconstruct(
        self,
        raw_stft: np.ndarray,
        debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整处理流程：去噪 + ISTFT重建
        
        Args:
            raw_stft: 原始STFT数据，形状 [2, 257, T]
            debug: 是否打印调试信息
            
        Returns:
            (denoised_stft, reconstructed_signal): 去噪后的STFT和重建的一维信号
        """
        # 1. 去噪
        denoised_stft = self.process_file(raw_stft, debug=debug)
        
        # 2. ISTFT重建
        reconstructed_signal = self.istft(denoised_stft)
        
        if debug:
            print(f"[DEBUG] ISTFT输出: length={len(reconstructed_signal)}, "
                  f"range=[{reconstructed_signal.min():.4f}, {reconstructed_signal.max():.4f}]")
        
        return denoised_stft, reconstructed_signal


def load_model(
    checkpoint_path: str,
    device: torch.device,
    base_channels: int = 32
) -> SpectrogramNAFNet:
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备
        base_channels: 基础通道数
        
    Returns:
        加载权重后的模型
    """
    model = SpectrogramNAFNet(
        in_channels=2,
        out_channels=2,
        base_channels=base_channels,
        num_blocks=[2, 2, 4, 8],
        bottleneck_blocks=4
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"模型加载成功: {checkpoint_path}")
    
    return model


def process_single_file(
    input_path: str,
    output_dir: str,
    processor: STFTInferenceProcessor,
    save_stft: bool = False,
    save_signal: bool = True
) -> Dict[str, str]:
    """
    处理单个文件
    
    Args:
        input_path: 输入STFT文件路径 (.npy)
        output_dir: 输出目录
        processor: 推理处理器
        save_stft: 是否保存去噪后的STFT
        save_signal: 是否保存重建的一维信号
        
    Returns:
        输出文件路径字典
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载输入数据
    raw_stft = np.load(input_path)
    print(f"加载文件: {input_path}")
    print(f"  输入形状: {raw_stft.shape}")
    
    # 处理
    denoised_stft, reconstructed_signal = processor.process_and_reconstruct(raw_stft)
    
    print(f"  去噪STFT形状: {denoised_stft.shape}")
    print(f"  重建信号长度: {len(reconstructed_signal)}")
    
    # 保存结果
    output_paths = {}
    base_name = input_path.stem.replace('_raw', '')
    
    if save_stft:
        stft_path = output_dir / f"{base_name}_denoised_stft.npy"
        np.save(stft_path, denoised_stft)
        output_paths['stft'] = str(stft_path)
        print(f"  保存STFT: {stft_path}")
    
    if save_signal:
        signal_path = output_dir / f"{base_name}_denoised_signal.npy"
        np.save(signal_path, reconstructed_signal)
        output_paths['signal'] = str(signal_path)
        print(f"  保存信号: {signal_path}")
    
    return output_paths


def batch_process(
    input_dir: str,
    output_dir: str,
    processor: STFTInferenceProcessor,
    pattern: str = '*_raw.npy'
) -> List[Dict[str, str]]:
    """
    批量处理文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        processor: 推理处理器
        pattern: 文件匹配模式
        
    Returns:
        所有输出文件路径列表
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    
    if len(files) == 0:
        print(f"未找到匹配文件: {input_dir / pattern}")
        return []
    
    print(f"找到 {len(files)} 个文件待处理")
    
    results = []
    for file_path in tqdm(files, desc="批量处理"):
        try:
            result = process_single_file(
                str(file_path),
                output_dir,
                processor
            )
            results.append(result)
        except Exception as e:
            print(f"处理失败: {file_path}, 错误: {e}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SpectrogramNAFNet 推理脚本'
    )
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='输入文件或目录路径'
    )
    parser.add_argument(
        '--output', type=str, default='inference_output',
        help='输出目录'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='批量处理模式'
    )
    parser.add_argument(
        '--pattern', type=str, default='*_raw.npy',
        help='批量模式下的文件匹配模式'
    )
    parser.add_argument(
        '--base_channels', type=int, default=32,
        help='模型基础通道数'
    )
    parser.add_argument(
        '--window_size', type=int, default=156,
        help='推理窗口大小'
    )
    parser.add_argument(
        '--overlap_ratio', type=float, default=0.5,
        help='窗口重叠比例 (0.5 表示 50%% 重叠)'
    )
    parser.add_argument(
        '--n_fft', type=int, default=512,
        help='STFT的FFT点数'
    )
    parser.add_argument(
        '--hop_length', type=int, default=64,
        help='STFT的跳跃长度'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=500,
        help='采样率'
    )
    parser.add_argument(
        '--no_stft', action='store_true',
        help='不保存去噪后的STFT'
    )
    parser.add_argument(
        '--no_signal', action='store_true',
        help='不保存重建的一维信号'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='计算设备 (auto/cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device, args.base_channels)
    
    # 创建处理器 (使用50%重叠 + Hann窗加权平均)
    processor = STFTInferenceProcessor(
        model=model,
        device=device,
        window_size=args.window_size,
        overlap_ratio=args.overlap_ratio,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate
    )
    
    print(f"推理参数: 窗口大小={args.window_size}, 重叠比例={args.overlap_ratio*100:.0f}%, 步长={processor.stride}")
    
    # 处理
    if args.batch:
        results = batch_process(
            args.input,
            args.output,
            processor,
            args.pattern
        )
        print(f"\n完成: 处理了 {len(results)} 个文件")
    else:
        result = process_single_file(
            args.input,
            args.output,
            processor,
            save_stft=not args.no_stft,
            save_signal=not args.no_signal
        )
        print(f"\n完成: {result}")


if __name__ == '__main__':
    main()
