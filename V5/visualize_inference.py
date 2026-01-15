"""
可视化推理结果
比较原始信号、真实clean信号（如果有）和降噪信号
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_comparison(
    raw_signal,
    denoised_signal,
    clean_signal=None,
    title="Signal Comparison",
    save_path=None,
    show_spectral=False,
    sample_rate=1000
):
    """
    绘制信号对比图
    
    Args:
        raw_signal: 原始噪声信号
        denoised_signal: 降噪后信号
        clean_signal: 真实clean信号（可选）
        title: 图表标题
        save_path: 保存路径
        show_spectral: 是否显示频谱图
        sample_rate: 采样率（Hz）
    """
    # 计算时间轴
    time = np.arange(len(raw_signal)) / sample_rate
    
    # 创建子图
    if show_spectral:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes = [axes]
    
    # 时域图
    ax = axes[0]
    ax.plot(time, raw_signal, 'k', alpha=0.4, linewidth=1, label='Raw (Noisy)')
    ax.plot(time, denoised_signal, 'r', linewidth=1.5, label='Denoised')
    
    if clean_signal is not None:
        ax.plot(time, clean_signal, 'g', linewidth=1.5, label='Ground Truth (Clean)')
        
        # 计算MSE
        mse_raw = np.mean((raw_signal - clean_signal) ** 2)
        mse_denoised = np.mean((denoised_signal - clean_signal) ** 2)
        improvement = (mse_raw - mse_denoised) / mse_raw * 100
        
        ax.set_title(
            f"{title}\n"
            f"Raw MSE: {mse_raw:.6f} | Denoised MSE: {mse_denoised:.6f} | "
            f"Improvement: {improvement:.1f}%",
            fontsize=12, fontweight='bold'
        )
    else:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 频谱图
    if show_spectral and len(axes) > 1:
        ax = axes[1]
        
        # 计算功率谱密度
        from scipy import signal as sp_signal
        
        f_raw, psd_raw = sp_signal.welch(raw_signal, fs=sample_rate, nperseg=1024)
        f_denoised, psd_denoised = sp_signal.welch(denoised_signal, fs=sample_rate, nperseg=1024)
        
        ax.semilogy(f_raw, psd_raw, 'k', alpha=0.4, linewidth=1, label='Raw (Noisy)')
        ax.semilogy(f_denoised, psd_denoised, 'r', linewidth=1.5, label='Denoised')
        
        if clean_signal is not None:
            f_clean, psd_clean = sp_signal.welch(clean_signal, fs=sample_rate, nperseg=1024)
            ax.semilogy(f_clean, psd_clean, 'g', linewidth=1.5, label='Ground Truth (Clean)')
        
        ax.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('PSD (V²/Hz)', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0, sample_rate / 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {save_path}")
    
    plt.show()


def plot_multiple_segments(
    raw_signals,
    denoised_signals,
    clean_signals=None,
    num_segments=4,
    title="Multiple Segments Comparison",
    save_path=None,
    sample_rate=1000
):
    """
    绘制多个片段的对比图
    
    Args:
        raw_signals: 原始信号列表
        denoised_signals: 降噪信号列表
        clean_signals: clean信号列表（可选）
        num_segments: 显示片段数量
        title: 图表标题
        save_path: 保存路径
        sample_rate: 采样率
    """
    num_segments = min(num_segments, len(raw_signals))
    
    fig, axes = plt.subplots(num_segments, 1, figsize=(15, 3 * num_segments))
    
    if num_segments == 1:
        axes = [axes]
    
    for i in range(num_segments):
        ax = axes[i]
        
        raw = raw_signals[i]
        denoised = denoised_signals[i]
        
        time = np.arange(len(raw)) / sample_rate
        
        ax.plot(time, raw, 'k', alpha=0.4, linewidth=1, label='Raw')
        ax.plot(time, denoised, 'r', linewidth=1.5, label='Denoised')
        
        if clean_signals is not None:
            clean = clean_signals[i]
            ax.plot(time, clean, 'g', linewidth=1.5, label='Clean')
            
            mse = np.mean((denoised - clean) ** 2)
            ax.set_title(f"Segment {i+1} - MSE: {mse:.6f}", fontsize=11, fontweight='bold')
        else:
            ax.set_title(f"Segment {i+1}", fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {save_path}")
    
    plt.show()


def load_and_visualize(
    raw_path,
    denoised_path,
    clean_path=None,
    segment_start=0,
    segment_length=2048,
    save_path=None,
    show_spectral=True
):
    """
    加载文件并可视化
    
    Args:
        raw_path: 原始信号文件路径
        denoised_path: 降噪信号文件路径
        clean_path: clean信号文件路径（可选）
        segment_start: 显示片段起始位置
        segment_length: 显示片段长度
        save_path: 保存路径
        show_spectral: 是否显示频谱
    """
    print(f"加载文件...")
    print(f"  Raw: {raw_path}")
    print(f"  Denoised: {denoised_path}")
    if clean_path:
        print(f"  Clean: {clean_path}")
    
    # 加载信号
    raw_signal = np.load(raw_path)
    denoised_signal = np.load(denoised_path)
    
    if raw_signal.ndim == 2:
        raw_signal = raw_signal[0, :]
    if denoised_signal.ndim == 2:
        denoised_signal = denoised_signal[0, :]
    
    clean_signal = None
    if clean_path:
        clean_signal = np.load(clean_path)
        if clean_signal.ndim == 2:
            clean_signal = clean_signal[0, :]
    
    # 截取片段
    if segment_length is not None:
        end = min(segment_start + segment_length, len(raw_signal))
        raw_signal = raw_signal[segment_start:end]
        denoised_signal = denoised_signal[segment_start:end]
        if clean_signal is not None:
            clean_signal = clean_signal[segment_start:end]
    
    print(f"\n信号长度: {len(raw_signal)} samples")
    
    # 绘图
    plot_comparison(
        raw_signal=raw_signal,
        denoised_signal=denoised_signal,
        clean_signal=clean_signal,
        title="Denoising Result Comparison",
        save_path=save_path,
        show_spectral=show_spectral
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize denoising results")
    
    parser.add_argument("--raw", type=str, required=True,
                        help="原始信号文件路径")
    parser.add_argument("--denoised", type=str, required=True,
                        help="降噪信号文件路径")
    parser.add_argument("--clean", type=str, default=None,
                        help="clean信号文件路径（可选）")
    
    parser.add_argument("--start", type=int, default=0,
                        help="显示片段起始位置")
    parser.add_argument("--length", type=int, default=2048,
                        help="显示片段长度（None为全长）")
    
    parser.add_argument("--save", type=str, default=None,
                        help="保存路径")
    parser.add_argument("--spectral", action="store_true",
                        help="显示频谱图")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("可视化推理结果")
    print("=" * 70)
    
    load_and_visualize(
        raw_path=args.raw,
        denoised_path=args.denoised,
        clean_path=args.clean,
        segment_start=args.start,
        segment_length=args.length,
        save_path=args.save,
        show_spectral=args.spectral
    )
    
    print("\n✓ 可视化完成!")


if __name__ == "__main__":
    main()
