"""
EEG数据集加载器
支持随机切片和Instance Normalization
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random


class EEGDataset(Dataset):
    """
    EEG去噪数据集
    
    每个样本包含一对(raw, clean) .npy文件
    训练时随机截取2048个连续采样点，并进行Instance Scaling
    
    Args:
        raw_paths: raw EEG文件路径列表
        clean_paths: clean EEG文件路径列表
        segment_length: 切片长度（默认2048，对应4秒@500Hz）
        augment: 是否进行数据增强（随机翻转等）
    """
    
    def __init__(
        self,
        raw_paths: List[str],
        clean_paths: List[str],
        segment_length: int = 2048,
        augment: bool = False
    ):
        assert len(raw_paths) == len(clean_paths), \
            f"Mismatch: {len(raw_paths)} raw files vs {len(clean_paths)} clean files"
        
        self.raw_paths = raw_paths
        self.clean_paths = clean_paths
        self.segment_length = segment_length
        self.augment = augment
        
        # 验证文件存在性
        self._validate_files()
        
        print(f"Loaded {len(self)} samples")
        print(f"Segment length: {segment_length} samples ({segment_length/500:.1f}s @ 500Hz)")
        print(f"Data augmentation: {augment}")
    
    def _validate_files(self):
        """验证所有文件是否存在"""
        for raw_path, clean_path in zip(self.raw_paths, self.clean_paths):
            if not Path(raw_path).exists():
                raise FileNotFoundError(f"Raw file not found: {raw_path}")
            if not Path(clean_path).exists():
                raise FileNotFoundError(f"Clean file not found: {clean_path}")
    
    def __len__(self) -> int:
        return len(self.raw_paths)
    
    def _instance_normalize(
        self,
        raw_segment: np.ndarray,
        clean_segment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Instance Normalization
        使用raw信号的统计量归一化raw和clean
        并显式强制均值为零，防止DC漂移
        
        Args:
            raw_segment: raw信号片段 [T]
            clean_segment: clean信号片段 [T]
            
        Returns:
            normalized_raw: 归一化后的raw [T]
            normalized_clean: 归一化后的clean [T]
        """
        # 计算raw的均值和标准差
        mean = np.mean(raw_segment)
        std = np.std(raw_segment)
        
        # 避免除零
        if std < 1e-8:
            std = 1.0
        
        # 归一化
        normalized_raw = (raw_segment - mean) / std
        normalized_clean = (clean_segment - mean) / std
        
        # 显式强制去除DC分量（确保均值为零）
        normalized_raw = normalized_raw - np.mean(normalized_raw)
        normalized_clean = normalized_clean - np.mean(normalized_clean)
        
        return normalized_raw, normalized_clean
    
    def _random_crop(
        self,
        raw_full: np.ndarray,
        clean_full: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机裁剪固定长度的片段
        
        Args:
            raw_full: 完整的raw信号 [T_full]
            clean_full: 完整的clean信号 [T_full]
            
        Returns:
            raw_crop: 裁剪后的raw [segment_length]
            clean_crop: 裁剪后的clean [segment_length]
        """
        total_length = len(raw_full)
        
        if total_length < self.segment_length:
            # 如果信号长度不足，进行零填充
            pad_length = self.segment_length - total_length
            raw_crop = np.pad(raw_full, (0, pad_length), mode='constant')
            clean_crop = np.pad(clean_full, (0, pad_length), mode='constant')
        elif total_length == self.segment_length:
            # 刚好等于目标长度
            raw_crop = raw_full
            clean_crop = clean_full
        else:
            # 随机选择起始位置
            max_start = total_length - self.segment_length
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.segment_length
            
            raw_crop = raw_full[start_idx:end_idx]
            clean_crop = clean_full[start_idx:end_idx]
        
        return raw_crop, clean_crop
    
    def _augment_data(
        self,
        raw: np.ndarray,
        clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强
        
        Args:
            raw: raw信号 [T]
            clean: clean信号 [T]
            
        Returns:
            augmented_raw: 增强后的raw [T]
            augmented_clean: 增强后的clean [T]
        """
        # 随机翻转（50%概率）
        if random.random() > 0.5:
            raw = -raw
            clean = -clean
        
        # 随机时间反转（50%概率）
        if random.random() > 0.5:
            raw = raw[::-1].copy()
            clean = clean[::-1].copy()
        
        return raw, clean
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            raw_tensor: 归一化的raw EEG [1, segment_length]
            clean_tensor: 归一化的clean EEG [1, segment_length]
        """
        # 加载完整的.npy文件
        raw_full = np.load(self.raw_paths[idx]).astype(np.float32)
        clean_full = np.load(self.clean_paths[idx]).astype(np.float32)
        
        # 确保是1D数组
        if raw_full.ndim > 1:
            raw_full = raw_full.squeeze()
        if clean_full.ndim > 1:
            clean_full = clean_full.squeeze()
        
        # 随机裁剪
        raw_crop, clean_crop = self._random_crop(raw_full, clean_full)
        
        # 数据增强
        if self.augment:
            raw_crop, clean_crop = self._augment_data(raw_crop, clean_crop)
        
        # Instance Normalization
        raw_norm, clean_norm = self._instance_normalize(raw_crop, clean_crop)
        
        # 转换为Tensor并添加通道维度
        raw_tensor = torch.from_numpy(raw_norm).unsqueeze(0)  # [1, T]
        clean_tensor = torch.from_numpy(clean_norm).unsqueeze(0)  # [1, T]
        
        return raw_tensor, clean_tensor


def build_dataset_from_directory(
    data_dir: str,
    split: str = 'train',
    segment_length: int = 2048,
    augment: bool = False
) -> EEGDataset:
    """
    从目录结构构建数据集
    
    期望的目录结构:
    data_dir/
        train/
            raw/
                subj001_raw.npy
                subj002_raw.npy
                ...
            clean/
                subj001_clean.npy
                subj002_clean.npy
                ...
        val/
            raw/
            clean/
        test/
            raw/
            clean/
    
    Args:
        data_dir: 数据根目录
        split: 数据集划分 ('train', 'val', 'test')
        segment_length: 切片长度
        augment: 是否数据增强（通常只在train时使用）
        
    Returns:
        dataset: EEGDataset实例
    """
    data_path = Path(data_dir) / split
    raw_dir = data_path / 'raw'
    clean_dir = data_path / 'clean'
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    
    # 获取所有.npy文件
    raw_files = sorted(raw_dir.glob('*.npy'))
    
    # 构建对应的clean文件路径
    raw_paths = []
    clean_paths = []
    
    for raw_file in raw_files:
        # 从raw文件名推断clean文件名
        # 假设命名约定: subjXXX_raw.npy -> subjXXX_clean.npy
        filename = raw_file.stem  # 'subjXXX_raw'
        
        if filename.endswith('_raw'):
            clean_filename = filename.replace('_raw', '_clean') + '.npy'
        else:
            # 如果不遵循约定，尝试直接替换
            clean_filename = filename + '.npy'
        
        clean_file = clean_dir / clean_filename
        
        if clean_file.exists():
            raw_paths.append(str(raw_file))
            clean_paths.append(str(clean_file))
        else:
            print(f"Warning: Clean file not found for {raw_file.name}, skipping...")
    
    print(f"\n=== Building {split} dataset ===")
    print(f"Data directory: {data_path}")
    print(f"Found {len(raw_paths)} matched pairs")
    
    return EEGDataset(
        raw_paths=raw_paths,
        clean_paths=clean_paths,
        segment_length=segment_length,
        augment=augment
    )


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试数据
    test_dir = Path("Dataset")
    
    if test_dir.exists():
        # 从真实数据测试
        try:
            dataset = build_dataset_from_directory(
                data_dir=str(test_dir),
                split='train',
                segment_length=2048,
                augment=True
            )
            
            print(f"\nDataset size: {len(dataset)}")
            
            # 获取一个样本
            raw, clean = dataset[0]
            
            print(f"\nSample shapes:")
            print(f"  Raw: {raw.shape}")
            print(f"  Clean: {clean.shape}")
            print(f"\nSample statistics:")
            print(f"  Raw - mean: {raw.mean():.4f}, std: {raw.std():.4f}")
            print(f"  Clean - mean: {clean.mean():.4f}, std: {clean.std():.4f}")
            
            # 可视化
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            
            time_axis = np.arange(2048) / 500  # 转换为秒
            
            axes[0].plot(time_axis, raw.squeeze().numpy(), linewidth=0.5)
            axes[0].set_title('Raw EEG (Noisy)')
            axes[0].set_ylabel('Amplitude (normalized)')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_axis, clean.squeeze().numpy(), linewidth=0.5)
            axes[1].set_title('Clean EEG')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Amplitude (normalized)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dataset_test.png', dpi=150)
            print("\nVisualization saved to 'dataset_test.png'")
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nCreating synthetic test data...")
    
    # 创建合成测试数据
    print("\n=== Creating synthetic test data ===")
    
    # 生成模拟信号
    fs = 500  # 采样率
    duration = 10  # 秒
    t = np.arange(0, duration, 1/fs)
    
    # Clean: 两个正弦波叠加
    clean_signal = (
        np.sin(2 * np.pi * 10 * t) +
        0.5 * np.sin(2 * np.pi * 20 * t)
    )
    
    # Raw: Clean + 噪声
    noise = 0.5 * np.random.randn(len(t))
    raw_signal = clean_signal + noise
    
    # 创建临时数据集
    raw_paths = ['temp_raw.npy']
    clean_paths = ['temp_clean.npy']
    
    np.save('temp_raw.npy', raw_signal.astype(np.float32))
    np.save('temp_clean.npy', clean_signal.astype(np.float32))
    
    dataset = EEGDataset(
        raw_paths=raw_paths,
        clean_paths=clean_paths,
        segment_length=2048,
        augment=True
    )
    
    # 测试多个样本（同一文件的不同切片）
    print("\nTesting random cropping:")
    for i in range(3):
        raw, clean = dataset[0]
        print(f"Sample {i+1}: Raw mean={raw.mean():.4f}, std={raw.std():.4f}")
    
    # 清理
    Path('temp_raw.npy').unlink()
    Path('temp_clean.npy').unlink()
    
    print("\n✓ Dataset test completed successfully!")
