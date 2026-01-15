"""
文件级推理脚本
支持单文件或目录批量推理，保存降噪后的时域信号
"""
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.uar_acssnet import UAR_ACSSNet
from configs.default import get_default_config


def load_model(checkpoint_path, device='cuda', baseline_mode=None):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
        baseline_mode: 是否使用baseline模式（None则从checkpoint获取）
    
    Returns:
        model: 加载好的模型
        cfg: 配置字典
    """
    print(f"\n📦 Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint获取配置（如果有）或使用默认配置
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
        print("✓ Using config from checkpoint")
    else:
        cfg = get_default_config()
        print("✓ Using default config")
    
    # 创建模型
    bm = cfg.get("baseline_mode", False) if baseline_mode is None else baseline_mode
    model = UAR_ACSSNet(
        segment_length=cfg.get("segment_length", 2048),
        unet_base_ch=cfg.get("unet_base_ch", 32),
        unet_levels=cfg.get("unet_levels", 4),
        spec_channels=cfg.get("spec_channels", 64),
        acss_depth=cfg.get("acss_depth", 3),
        num_freq_bins=cfg.get("num_freq_bins", 101),
        dropout=cfg.get("dropout", 0.0),
        baseline_mode=bm,
    ).to(device)
    
    # 加载权重
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
    val_loss = ckpt.get('val_loss', None)
    if isinstance(val_loss, (float, int)):
        print(f"✓ Best val loss: {val_loss:.6f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Mode: {'Baseline U-Net' if model.baseline_mode else 'Full UAR-ACSSNet'}")
    
    return model, cfg


def normalize_signal(signal, method='zscore'):
    """
    归一化信号
    
    Args:
        signal: 输入信号 (numpy array)
        method: 归一化方法 ('zscore', 'minmax', 'none')
    
    Returns:
        normalized_signal: 归一化后的信号
        stats: 统计信息字典（用于反归一化）
    """
    if method == 'zscore':
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-8:
            std = 1.0
        normalized = (signal - mean) / std
        stats = {'mean': mean, 'std': std, 'method': 'zscore'}
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val < 1e-8:
            normalized = signal
        else:
            normalized = (signal - min_val) / (max_val - min_val)
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:  # 'none'
        normalized = signal
        stats = {'method': 'none'}
    
    return normalized, stats


def denormalize_signal(signal, stats):
    """
    反归一化信号
    
    Args:
        signal: 归一化后的信号
        stats: 归一化统计信息
    
    Returns:
        原始尺度的信号
    """
    method = stats.get('method', 'none')
    
    if method == 'zscore':
        return signal * stats['std'] + stats['mean']
    elif method == 'minmax':
        return signal * (stats['max'] - stats['min']) + stats['min']
    else:
        return signal


def segment_signal(signal, segment_length, stride):
    """
    将长信号分割成多个片段
    
    Args:
        signal: 输入信号 (N,) 或 (1, N)
        segment_length: 片段长度
        stride: 滑窗步长
    
    Returns:
        segments: (num_segments, 1, segment_length)
        num_segments: 片段数量
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]  # (1, N)
    
    n_samples = signal.shape[1]
    
    if n_samples <= segment_length:
        # 信号长度不足，填充
        padded = np.zeros((1, segment_length))
        padded[:, :n_samples] = signal
        return padded[np.newaxis, :, :], 1, n_samples
    
    # 滑窗分割
    segments = []
    start = 0
    while start + segment_length <= n_samples:
        segment = signal[:, start:start + segment_length]
        segments.append(segment)
        start += stride
    
    # 处理最后一个片段（如果需要）
    if start < n_samples:
        last_segment = np.zeros((1, segment_length))
        remaining = n_samples - start
        last_segment[:, :remaining] = signal[:, start:]
        segments.append(last_segment)
    
    segments = np.stack(segments, axis=0)  # (num_segments, 1, segment_length)
    return segments, len(segments), n_samples


def reconstruct_signal(segments, original_length, stride):
    """
    从分割的片段重建完整信号（使用重叠平均）
    
    Args:
        segments: (num_segments, 1, segment_length)
        original_length: 原始信号长度
        stride: 滑窗步长
    
    Returns:
        reconstructed: (original_length,)
    """
    num_segments, _, segment_length = segments.shape
    
    # 如果只有一个片段
    if num_segments == 1:
        return segments[0, 0, :original_length]
    
    # 重叠平均重建
    reconstructed = np.zeros(original_length)
    counts = np.zeros(original_length)
    
    start = 0
    for i in range(num_segments):
        end = min(start + segment_length, original_length)
        length = end - start
        reconstructed[start:end] += segments[i, 0, :length]
        counts[start:end] += 1
        start += stride
    
    # 避免除零
    counts = np.maximum(counts, 1)
    reconstructed = reconstructed / counts
    
    return reconstructed


def inference_single_file(
    model,
    input_path,
    output_path,
    device='cuda',
    segment_length=2048,
    stride=1024,
    normalize='zscore',
    batch_size=32,
    save_format='npy'
):
    """
    对单个文件进行推理
    
    Args:
        model: 模型
        input_path: 输入文件路径 (.npy)
        output_path: 输出文件路径
        device: 设备
        segment_length: 分割长度
        stride: 滑窗步长
        normalize: 归一化方法
        batch_size: 批处理大小
        save_format: 保存格式 ('npy', 'npz', 'txt')
    
    Returns:
        stats: 推理统计信息
    """
    # 加载信号
    signal = np.load(input_path)
    if signal.ndim == 2:
        signal = signal[0, :]  # 取第一个通道
    
    original_length = len(signal)
    
    # 归一化
    signal_norm, norm_stats = normalize_signal(signal, method=normalize)
    
    # 分割
    segments, num_segments, actual_length = segment_signal(
        signal_norm, segment_length, stride
    )
    
    # 批量推理
    denoised_segments = []
    
    with torch.no_grad():
        for i in range(0, num_segments, batch_size):
            batch = segments[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            outputs = model(batch_tensor)
            y_hat = outputs['y_hat'].cpu().numpy()
            
            denoised_segments.append(y_hat)
    
    denoised_segments = np.concatenate(denoised_segments, axis=0)
    
    # 重建完整信号
    denoised_signal = reconstruct_signal(
        denoised_segments, actual_length, stride
    )
    
    # 反归一化
    denoised_signal = denormalize_signal(denoised_signal, norm_stats)
    
    # 截取到原始长度
    denoised_signal = denoised_signal[:original_length]
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    if save_format == 'npy':
        np.save(output_path, denoised_signal)
    elif save_format == 'npz':
        np.savez(
            output_path,
            denoised=denoised_signal,
            original=signal,
            metadata={'num_segments': num_segments, 'stride': stride}
        )
    elif save_format == 'txt':
        np.savetxt(output_path, denoised_signal)
    
    # 计算统计信息
    mse = np.mean((signal - denoised_signal) ** 2)
    
    stats = {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'original_length': original_length,
        'num_segments': num_segments,
        'mse': float(mse),
        'signal_std': float(np.std(signal)),
        'denoised_std': float(np.std(denoised_signal))
    }
    
    return stats


def inference_directory(
    model,
    input_dir,
    output_dir,
    device='cuda',
    segment_length=2048,
    stride=1024,
    normalize='zscore',
    batch_size=32,
    save_format='npy',
    pattern='*.npy'
):
    """
    对目录内所有文件进行批量推理
    
    Args:
        model: 模型
        input_dir: 输入目录
        output_dir: 输出目录
        device: 设备
        segment_length: 分割长度
        stride: 滑窗步长
        normalize: 归一化方法
        batch_size: 批处理大小
        save_format: 保存格式
        pattern: 文件匹配模式
    
    Returns:
        all_stats: 所有文件的统计信息列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 查找所有匹配的文件
    input_files = sorted(input_dir.glob(pattern))
    
    if len(input_files) == 0:
        print(f"⚠️  No files found matching pattern '{pattern}' in {input_dir}")
        return []
    
    print(f"\n🔍 Found {len(input_files)} files to process")
    
    all_stats = []
    
    # 处理每个文件
    for input_path in tqdm(input_files, desc="Processing files"):
        try:
            # 构建输出路径
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path.stem
            
            if save_format == 'npy':
                output_path = output_path.with_suffix('.npy')
            elif save_format == 'npz':
                output_path = output_path.with_suffix('.npz')
            elif save_format == 'txt':
                output_path = output_path.with_suffix('.txt')
            
            # 推理
            stats = inference_single_file(
                model=model,
                input_path=input_path,
                output_path=output_path,
                device=device,
                segment_length=segment_length,
                stride=stride,
                normalize=normalize,
                batch_size=batch_size,
                save_format=save_format
            )
            
            all_stats.append(stats)
            
        except Exception as e:
            print(f"\n❌ Error processing {input_path.name}: {str(e)}")
            continue
    
    # 保存统计信息
    stats_path = output_dir / 'inference_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✓ Processed {len(all_stats)} files")
    print(f"✓ Statistics saved to: {stats_path}")
    
    # 打印汇总统计
    if all_stats:
        avg_mse = np.mean([s['mse'] for s in all_stats])
        print(f"\n📊 Summary:")
        print(f"  - Average MSE: {avg_mse:.6f}")
        print(f"  - Total files: {len(all_stats)}")
    
    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(description="File-level inference for EEG denoising")
    
    # 模型配置
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    
    # 输入/输出配置
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件或目录路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件或目录路径")
    parser.add_argument("--pattern", type=str, default="*.npy",
                        help="文件匹配模式（仅目录模式）")
    
    # 推理参数
    parser.add_argument("--segment_length", type=int, default=2048,
                        help="分割长度")
    parser.add_argument("--stride", type=int, default=1024,
                        help="滑窗步长（建议为segment_length的一半）")
    parser.add_argument("--normalize", type=str, default="zscore",
                        choices=['zscore', 'minmax', 'none'],
                        help="归一化方法")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--save_format", type=str, default="npy",
                        choices=['npy', 'npz', 'txt'],
                        help="保存格式")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 打印配置
    print("=" * 70)
    print("EEG Denoising - File-level Inference")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Segment length: {args.segment_length}")
    print(f"Stride: {args.stride}")
    print(f"Normalize: {args.normalize}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save format: {args.save_format}")
    print("=" * 70)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
    
    # 加载模型
    model, cfg = load_model(args.checkpoint, device=device)
    
    # 判断是文件还是目录
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单文件推理
        print("\n📄 Single file mode")
        stats = inference_single_file(
            model=model,
            input_path=input_path,
            output_path=args.output,
            device=device,
            segment_length=args.segment_length,
            stride=args.stride,
            normalize=args.normalize,
            batch_size=args.batch_size,
            save_format=args.save_format
        )
        
        print("\n✓ Inference completed!")
        print(f"  - Input: {stats['input_file']}")
        print(f"  - Output: {stats['output_file']}")
        print(f"  - Original length: {stats['original_length']:,} samples")
        print(f"  - Num segments: {stats['num_segments']}")
        print(f"  - MSE: {stats['mse']:.6f}")
        
    elif input_path.is_dir():
        # 目录批量推理
        print("\n📁 Directory mode")
        all_stats = inference_directory(
            model=model,
            input_dir=input_path,
            output_dir=args.output,
            device=device,
            segment_length=args.segment_length,
            stride=args.stride,
            normalize=args.normalize,
            batch_size=args.batch_size,
            save_format=args.save_format,
            pattern=args.pattern
        )
        
        print("\n✓ Batch inference completed!")
        
    else:
        print(f"❌ Error: Input path '{input_path}' does not exist")
        return
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
