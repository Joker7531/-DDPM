"""
批量推理脚本 - 完整1000步DDPM采样
处理Dataset中所有的raw文件（train/val/test）
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
from tqdm import tqdm
import warnings
import time
from datetime import datetime

from model import ConditionalDiffWave
from diffusion import GaussianDiffusion


class FullStepBatchDenoiser:
    """
    批量去噪推理器 - 完整1000步DDPM采样
    
    Args:
        model_path: 模型检查点路径
        device: 推理设备
        segment_length: 片段长度（必须与训练时一致）
        hop_length: 跳跃长度（窗口重叠控制）
        use_amp: 是否使用混合精度加速推理
        baseline_correction: 是否进行基线校正
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        segment_length: int = 2048,
        hop_length: int = 1024,
        use_amp: bool = True,
        baseline_correction: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.use_amp = use_amp
        self.baseline_correction = baseline_correction
        
        print("="*80)
        print("批量推理配置 (完整1000步DDPM)")
        print("="*80)
        print(f"模型路径: {model_path}")
        print(f"推理设备: {self.device}")
        print(f"片段长度: {segment_length} samples ({segment_length/500:.2f}s @ 500Hz)")
        print(f"跳跃长度: {hop_length} samples (重叠: {(1 - hop_length/segment_length)*100:.1f}%)")
        print(f"采样步数: 1000 (完整DDPM，无加速)")
        print(f"混合精度: {use_amp}")
        print(f"基线校正: {baseline_correction}")
        print("="*80)
        print()
        
        # 创建模型
        print("加载模型...")
        model = ConditionalDiffWave(
            in_channels=2,
            out_channels=1,
            residual_channels=256,
            num_layers=30,
            dilation_cycle=10,
            time_emb_dim=512
        ).to(self.device)
        
        # 注意：这里sampling_timesteps=None表示使用完整的1000步
        self.diffusion = GaussianDiffusion(
            model=model,
            timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            loss_type='hybrid',
            sampling_timesteps=None  # 使用完整timesteps
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.diffusion.eval()
        
        # 创建Hann窗口用于overlap-add
        self.window = torch.hann_window(segment_length).to(self.device)
        
        print("✓ 模型加载成功!")
        if 'epoch' in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  验证损失: {checkpoint['val_loss']:.6f}")
        print()
    
    def _normalize_segment(self, segment: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Instance Normalization"""
        mean = np.mean(segment)
        std = np.std(segment)
        
        if std < 1e-8:
            std = 1.0
        
        normalized = (segment - mean) / std
        return normalized, mean, std
    
    def _denormalize_segment(self, segment: np.ndarray, mean: float, std: float) -> np.ndarray:
        """反归一化"""
        return segment * std + mean
    
    def _baseline_correct(self, signal: np.ndarray) -> np.ndarray:
        """基线校正：移除DC分量"""
        return signal - np.mean(signal)
    
    @torch.no_grad()
    def denoise_segment(self, raw_segment: np.ndarray) -> np.ndarray:
        """
        对单个片段进行去噪（完整1000步）
        
        Args:
            raw_segment: 原始片段 [segment_length]
            
        Returns:
            denoised_segment: 去噪后的片段 [segment_length]
        """
        # 归一化
        normalized, mean, std = self._normalize_segment(raw_segment)
        
        # 转换为tensor
        condition = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 推理（完整1000步，不使用DDIM加速）
        if self.use_amp:
            with torch.cuda.amp.autocast():
                denoised = self.diffusion.sample(
                    condition, 
                    ddim_sampling=False,  # 不使用DDIM，完整采样
                    show_progress=False   # 关闭单个片段的进度条
                )
        else:
            denoised = self.diffusion.sample(
                condition,
                ddim_sampling=False,
                show_progress=False
            )
        
        # 转换回numpy
        denoised_np = denoised.squeeze().cpu().numpy()
        
        # 反归一化
        denoised_denorm = self._denormalize_segment(denoised_np, mean, std)
        
        # 基线校正
        if self.baseline_correction:
            denoised_denorm = self._baseline_correct(denoised_denorm)
        
        return denoised_denorm
    
    def denoise_full_signal(self, raw_signal: np.ndarray, show_progress: bool = True) -> np.ndarray:
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
                f"信号长度 ({signal_length}) 短于片段长度 ({self.segment_length})。使用零填充。"
            )
            padded = np.pad(raw_signal, (0, self.segment_length - signal_length), mode='constant')
            denoised = self.denoise_segment(padded)
            return denoised[:signal_length]
        
        # 初始化输出缓冲区
        output = np.zeros(signal_length, dtype=np.float32)
        weights = np.zeros(signal_length, dtype=np.float32)
        
        # 计算窗口数量
        num_windows = int(np.ceil((signal_length - self.segment_length) / self.hop_length)) + 1
        
        # Hann窗口
        window_np = np.hanning(self.segment_length).astype(np.float32)
        
        # 滑动窗口处理
        iterator = range(num_windows)
        if show_progress:
            iterator = tqdm(iterator, desc='  处理片段', unit='window', leave=False)
        
        for i in iterator:
            start = i * self.hop_length
            end = start + self.segment_length
            
            # 处理最后一个窗口
            if end > signal_length:
                start = signal_length - self.segment_length
                end = signal_length
                
                if start < (i - 1) * self.hop_length + self.segment_length:
                    continue
            
            # 提取并去噪
            segment = raw_signal[start:end]
            denoised_segment = self.denoise_segment(segment)
            
            # 应用窗口函数
            denoised_windowed = denoised_segment * window_np
            
            # Overlap-add
            output[start:end] += denoised_windowed
            weights[start:end] += window_np
        
        # 归一化
        weights = np.maximum(weights, 1e-8)
        output = output / weights
        
        # 最终基线校正
        if self.baseline_correction:
            output = self._baseline_correct(output)
        
        return output
    
    def process_file(self, raw_path: Path, output_path: Path) -> bool:
        """
        处理单个文件
        
        Args:
            raw_path: 原始文件路径
            output_path: 输出文件路径
            
        Returns:
            success: 是否成功处理
        """
        try:
            # 加载信号
            raw_signal = np.load(raw_path).astype(np.float32)
            
            if raw_signal.ndim > 1:
                raw_signal = raw_signal.squeeze()
            
            # 去噪
            start_time = time.time()
            denoised_signal = self.denoise_full_signal(raw_signal, show_progress=True)
            elapsed = time.time() - start_time
            
            # 保存
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, denoised_signal)
            
            # 统计信息
            signal_duration = len(raw_signal) / 500  # 500Hz
            print(f"    ✓ 完成 | 信号: {signal_duration:.1f}s | 耗时: {elapsed:.1f}s | 速度比: {signal_duration/elapsed:.2f}x")
            
            return True
            
        except Exception as e:
            print(f"    ✗ 错误: {e}")
            return False
    
    def batch_process_dataset(
        self,
        dataset_dir: str,
        output_dir: str,
        subsets: List[str] = ['train', 'val', 'test']
    ):
        """
        批量处理Dataset中的所有raw文件
        
        Args:
            dataset_dir: Dataset目录路径
            output_dir: 输出目录路径
            subsets: 要处理的子集列表
        """
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        
        print("="*80)
        print("开始批量推理")
        print("="*80)
        print(f"数据集路径: {dataset_path}")
        print(f"输出路径: {output_path}")
        print(f"处理子集: {subsets}")
        print()
        
        # 收集所有需要处理的文件
        all_files = []
        for subset in subsets:
            raw_dir = dataset_path / subset / 'raw'
            if not raw_dir.exists():
                print(f"⚠ 警告: {raw_dir} 不存在，跳过")
                continue
            
            # 只处理非segment文件（避免重复处理切片数据）
            files = [f for f in raw_dir.glob('*.npy') if 'segment' not in f.name]
            all_files.extend([(f, subset) for f in files])
        
        if len(all_files) == 0:
            print("❌ 未找到任何文件！")
            return
        
        print(f"📊 找到 {len(all_files)} 个文件待处理\n")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        total_start_time = time.time()
        
        # 处理每个文件
        for idx, (raw_path, subset) in enumerate(all_files, 1):
            print(f"[{idx}/{len(all_files)}] {subset}/{raw_path.name}")
            
            # 构造输出路径（保持相同的目录结构）
            output_file_path = output_path / subset / 'denoised' / raw_path.name.replace('_raw.npy', '_denoised.npy')
            
            # 处理文件
            success = self.process_file(raw_path, output_file_path)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            print()
        
        # 总结
        total_elapsed = time.time() - total_start_time
        print("="*80)
        print("批量推理完成")
        print("="*80)
        print(f"总文件数: {len(all_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"总耗时: {total_elapsed/60:.1f} 分钟")
        print(f"平均每文件: {total_elapsed/len(all_files):.1f} 秒")
        print(f"结果保存至: {output_path}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='批量EEG信号去噪推理 - 完整1000步DDPM采样'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='模型检查点路径 (.pt file)'
    )
    parser.add_argument(
        '--dataset_dir', 
        type=str, 
        default='../Dataset',
        help='Dataset目录路径 (默认: ../Dataset)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../Dataset_denoised_full1000',
        help='输出目录路径 (默认: ../Dataset_denoised_full1000)'
    )
    parser.add_argument(
        '--subsets', 
        type=str, 
        nargs='+', 
        default=['train', 'val', 'test'],
        help='要处理的子集 (默认: train val test)'
    )
    parser.add_argument(
        '--segment_length', 
        type=int, 
        default=2048,
        help='片段长度，必须与训练时一致 (默认: 2048)'
    )
    parser.add_argument(
        '--hop_length', 
        type=int, 
        default=1024,
        help='跳跃长度，控制overlap (默认: 1024, 50%%重叠)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='推理设备 (默认: cuda)'
    )
    parser.add_argument(
        '--no_amp', 
        action='store_true',
        help='禁用混合精度'
    )
    parser.add_argument(
        '--no_baseline_correction', 
        action='store_true',
        help='禁用基线校正'
    )
    
    args = parser.parse_args()
    
    # 打印开始时间
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 创建去噪器
    denoiser = FullStepBatchDenoiser(
        model_path=args.model,
        device=args.device,
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        use_amp=not args.no_amp,
        baseline_correction=not args.no_baseline_correction
    )
    
    # 执行批量推理
    denoiser.batch_process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        subsets=args.subsets
    )
    
    # 打印结束时间
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
