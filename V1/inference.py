"""
推理脚本 - 使用Overlap-Add方法处理完整EEG信号
支持单文件推理和批量推理
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from scipy import signal

from model import ConditionalDiffWave
from diffusion import GaussianDiffusion


class EEGDenoiser:
    """
    EEG去噪推理器
    
    使用overlap-add方法处理长信号
    
    Args:
        model_path: 模型检查点路径
        device: 推理设备
        segment_length: 片段长度（必须与训练时一致）
        hop_length: 跳跃长度（窗口重叠控制）
        use_amp: 是否使用混合精度加速推理
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        segment_length: int = 2048,
        hop_length: int = 1024,  # 50% 重叠
        use_amp: bool = True,
        baseline_correction: bool = True,
        highpass_freq: float = 0.5,  # 高通滤波截止频率 (Hz)
        sampling_timesteps: int = 100  # 快速采样步数（默认100步，原始1000步）
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.use_amp = use_amp
        self.baseline_correction = baseline_correction
        self.highpass_freq = highpass_freq
        self.sampling_timesteps = sampling_timesteps
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Segment length: {segment_length} samples ({segment_length/500:.2f}s @ 500Hz)")
        print(f"Hop length: {hop_length} samples (overlap: {(1 - hop_length/segment_length)*100:.1f}%)")
        print(f"Sampling timesteps: {sampling_timesteps} (accelerated from 1000)")
        print(f"Baseline correction: {baseline_correction}")
        if baseline_correction and highpass_freq > 0:
            print(f"Highpass filter: {highpass_freq} Hz")
        
        # 创建模型
        model = ConditionalDiffWave(
            in_channels=2,
            out_channels=1,
            residual_channels=256,
            num_layers=30,
            dilation_cycle=10,
            time_emb_dim=512
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
            model=model,
            timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            loss_type='hybrid',
            sampling_timesteps=sampling_timesteps
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.diffusion.eval()
        
        # 创建Hann窗口用于overlap-add
        self.window = torch.hann_window(segment_length).to(self.device)
        
        print(f"Model loaded successfully!")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
        print()
    
    def _normalize_segment(
        self,
        segment: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        归一化片段（Instance Normalization）
        
        Args:
            segment: 输入片段 [T]
            
        Returns:
            normalized: 归一化后的片段
            mean: 均值
            std: 标准差
        """
        mean = np.mean(segment)
        std = np.std(segment)
        
        if std < 1e-8:
            std = 1.0
        
        normalized = (segment - mean) / std
        
        return normalized, mean, std
    
    def _denormalize_segment(
        self,
        segment: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """
        反归一化片段
        
        Args:
            segment: 归一化的片段 [T]
            mean: 原始均值
            std: 原始标准差
            
        Returns:
            denormalized: 反归一化后的片段
        """
        return segment * std + mean
    
    def _baseline_correct(self, signal: np.ndarray) -> np.ndarray:
        """
        基线校正：移除信号的DC分量（均值）
        
        Args:
            signal: 输入信号 [T]
            
        Returns:
            corrected: 基线校正后的信号 [T]
        """
        return signal - np.mean(signal)
    
    def _highpass_filter(self, x: np.ndarray, fs: float = 500.0) -> np.ndarray:
        """
        高通滤波：移除低频漂移
        
        Args:
            x: 输入信号 [T]
            fs: 采样率 (Hz)
            
        Returns:
            filtered: 滤波后的信号 [T]
        """
        if self.highpass_freq <= 0:
            return x
        
        # 设计Butterworth高通滤波器
        nyq = 0.5 * fs
        normal_cutoff = self.highpass_freq / nyq
        
        # 使用4阶Butterworth滤波器
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # 使用filtfilt进行零相位滤波
        filtered = signal.filtfilt(b, a, x)
        
        return filtered
    
    @torch.no_grad()
    def denoise_segment(
        self,
        raw_segment: np.ndarray,
        return_input_stats: bool = False
    ) -> np.ndarray:
        """
        对单个片段进行去噪
        
        Args:
            raw_segment: 原始片段 [segment_length]
            return_input_stats: 是否返回输入统计量
            
        Returns:
            denoised_segment: 去噪后的片段 [segment_length]
        """
        # 归一化
        normalized, mean, std = self._normalize_segment(raw_segment)
        
        # 转换为tensor
        condition = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 推理
        if self.use_amp:
            with torch.cuda.amp.autocast():
                denoised = self.diffusion.sample(
                    condition, 
                    ddim_sampling=True,
                    show_progress=False  # 关闭进度条以加速
                )
        else:
            denoised = self.diffusion.sample(
                condition,
                ddim_sampling=True,
                show_progress=False
            )
        
        # 转换回numpy
        denoised_np = denoised.squeeze().cpu().numpy()
        
        # 反归一化
        denoised_denorm = self._denormalize_segment(denoised_np, mean, std)
        
        # 基线校正：强制拉回零基线
        if self.baseline_correction:
            denoised_denorm = self._baseline_correct(denoised_denorm)
        
        if return_input_stats:
            return denoised_denorm, mean, std
        else:
            return denoised_denorm
    
    def denoise_full_signal(
        self,
        raw_signal: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        使用overlap-add方法对完整信号进行去噪
        
        Args:
            raw_signal: 完整的原始信号 [T]
            show_progress: 是否显示进度条
            
        Returns:
            denoised_signal: 去噪后的完整信号 [T]
        """
        signal_length = len(raw_signal)
        
        if signal_length < self.segment_length:
            warnings.warn(
                f"Signal length ({signal_length}) is shorter than segment length ({self.segment_length}). "
                f"Padding with zeros."
            )
            # 零填充
            padded = np.pad(raw_signal, (0, self.segment_length - signal_length), mode='constant')
            denoised = self.denoise_segment(padded)
            return denoised[:signal_length]
        
        # 初始化输出缓冲区和权重缓冲区
        output = np.zeros(signal_length, dtype=np.float32)
        weights = np.zeros(signal_length, dtype=np.float32)
        
        # 计算需要处理的窗口数量
        num_windows = int(np.ceil((signal_length - self.segment_length) / self.hop_length)) + 1
        
        # Hann窗口（numpy版本）
        window_np = np.hanning(self.segment_length).astype(np.float32)
        
        # 滑动窗口处理
        iterator = range(num_windows)
        if show_progress:
            iterator = tqdm(iterator, desc='Denoising', unit='window')
        
        for i in iterator:
            # 计算窗口起始位置
            start = i * self.hop_length
            end = start + self.segment_length
            
            # 处理最后一个窗口（可能超出信号长度）
            if end > signal_length:
                # 从信号末尾向前取segment_length个点
                start = signal_length - self.segment_length
                end = signal_length
                
                # 如果这个窗口已经被处理过（重叠区域），则跳过
                if start < (i - 1) * self.hop_length + self.segment_length:
                    continue
            
            # 提取片段
            segment = raw_signal[start:end]
            
            # 去噪
            denoised_segment = self.denoise_segment(segment)
            
            # 应用窗口函数（减少边界效应）
            denoised_windowed = denoised_segment * window_np
            
            # Overlap-add
            output[start:end] += denoised_windowed
            weights[start:end] += window_np
        
        # 归一化（处理重叠区域）
        # 避免除零
        weights = np.maximum(weights, 1e-8)
        output = output / weights
        
        # 最终基线校正
        if self.baseline_correction:
            output = self._baseline_correct(output)
            
            # 可选：应用高通滤波器进一步移除低频漂移
            if self.highpass_freq > 0:
                output = self._highpass_filter(output, fs=500.0)
        
        return output
    
    def denoise_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        save_comparison: bool = True
    ) -> np.ndarray:
        """
        对.npy文件进行去噪
        
        Args:
            input_path: 输入.npy文件路径
            output_path: 输出.npy文件路径（可选，默认在同目录下添加_denoised后缀）
            save_comparison: 是否保存对比图
            
        Returns:
            denoised_signal: 去噪后的信号
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Processing: {input_path.name}")
        
        # 加载信号
        raw_signal = np.load(input_path).astype(np.float32)
        
        if raw_signal.ndim > 1:
            raw_signal = raw_signal.squeeze()
        
        print(f"Signal length: {len(raw_signal)} samples ({len(raw_signal)/500:.2f}s @ 500Hz)")
        
        # 去噪
        denoised_signal = self.denoise_full_signal(raw_signal, show_progress=True)
        
        # 保存结果
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_denoised.npy"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, denoised_signal)
        print(f"Saved to: {output_path}")
        
        # 保存对比图
        if save_comparison:
            comparison_path = output_path.parent / f"{output_path.stem}_comparison.png"
            self._save_comparison_plot(raw_signal, denoised_signal, comparison_path)
            print(f"Comparison plot saved to: {comparison_path}")
        
        return denoised_signal
    
    def _save_comparison_plot(
        self,
        raw: np.ndarray,
        denoised: np.ndarray,
        save_path: Path,
        max_samples: int = 10000  # 最多显示10000个采样点
    ):
        """保存对比图"""
        # 如果信号太长，只显示前max_samples个点
        if len(raw) > max_samples:
            raw_display = raw[:max_samples]
            denoised_display = denoised[:max_samples]
            title_suffix = f" (showing first {max_samples/500:.1f}s)"
        else:
            raw_display = raw
            denoised_display = denoised
            title_suffix = ""
        
        time_axis = np.arange(len(raw_display)) / 500  # 500Hz
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Raw signal
        axes[0].plot(time_axis, raw_display, linewidth=0.5, color='#E74C3C', alpha=0.8)
        axes[0].set_title(f'Raw Signal (Noisy){title_suffix}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Denoised signal
        axes[1].plot(time_axis, denoised_display, linewidth=0.5, color='#3498DB', alpha=0.8)
        axes[1].set_title('Denoised Signal', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Difference (noise removed)
        difference = raw_display - denoised_display
        axes[2].plot(time_axis, difference, linewidth=0.5, color='#95A5A6', alpha=0.8)
        axes[2].set_title('Removed Noise (Raw - Denoised)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=10)
        axes[2].set_ylabel('Amplitude', fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_denoise(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = '*.npy',
        save_comparison: bool = True
    ):
        """
        批量处理目录中的所有文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            save_comparison: 是否保存对比图
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有匹配的文件
        files = list(input_path.glob(pattern))
        
        if len(files) == 0:
            print(f"No files found matching pattern '{pattern}' in {input_dir}")
            return
        
        print(f"Found {len(files)} files to process\n")
        
        # 处理每个文件
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            
            output_file = output_path / f"{file.stem}_denoised.npy"
            
            try:
                self.denoise_file(
                    input_path=str(file),
                    output_path=str(output_file),
                    save_comparison=save_comparison
                )
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='EEG Signal Denoising Inference')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file or directory (optional)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process all .npy files in input directory')
    parser.add_argument('--pattern', type=str, default='*_raw.npy',
                        help='File pattern for batch mode (default: *_raw.npy)')
    parser.add_argument('--segment_length', type=int, default=2048,
                        help='Segment length (must match training)')
    parser.add_argument('--hop_length', type=int, default=1024,
                        help='Hop length for overlap-add (default: 1024, 50%% overlap)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Do not save comparison plots')
    parser.add_argument('--no_baseline_correction', action='store_true',
                        help='Disable baseline correction (DC removal)')
    parser.add_argument('--highpass_freq', type=float, default=0.5,
                        help='Highpass filter cutoff frequency in Hz (default: 0.5, 0 to disable)')
    parser.add_argument('--sampling_timesteps', type=int, default=100,
                        help='Number of sampling timesteps for fast inference (default: 100, max: 1000)')
    
    args = parser.parse_args()
    
    # 创建去噪器
    denoiser = EEGDenoiser(
        model_path=args.model,
        device=args.device,
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        use_amp=not args.no_amp,
        baseline_correction=not args.no_baseline_correction,
        highpass_freq=args.highpass_freq,
        sampling_timesteps=args.sampling_timesteps
    )
    
    # 执行推理
    if args.batch:
        # 批量模式
        denoiser.batch_denoise(
            input_dir=args.input,
            output_dir=args.output or f"{args.input}_denoised",
            pattern=args.pattern,
            save_comparison=not args.no_comparison
        )
    else:
        # 单文件模式
        denoiser.denoise_file(
            input_path=args.input,
            output_path=args.output,
            save_comparison=not args.no_comparison
        )


if __name__ == '__main__':
    main()
