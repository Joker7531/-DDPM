"""
STFT域单通道EEG信号去噪数据集模块

本模块实现了基于滑动窗口的STFT频谱切片数据集，
支持残差学习策略，用于训练SpectrogramNAFNet模型。

作者: AI Assistant
日期: 2025-12-30
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class STFTSlicingDataset(Dataset):
    """
    STFT域滑动窗口切片数据集
    
    该数据集从STFT频谱图中提取固定大小的切片，
    支持频域裁剪、抗恒等映射归一化和残差目标计算。
    
    Attributes:
        data_dir: 数据目录路径
        window_size: 滑动窗口大小（帧数）
        stride: 滑动步长
        freq_start: 频率起始索引
        freq_end: 频率结束索引
        eps: 数值稳定性常数
    """
    
    # 频域裁剪参数
    FREQ_START: int = 1
    FREQ_END: int = 104  # 索引 1 到 103 (不含104)
    
    # 窗口参数
    WINDOW_SIZE: int = 156
    
    def __init__(
        self,
        raw_dir: Union[str, Path],
        clean_dir: Union[str, Path],
        mode: str = 'train',
        window_size: int = 156,
        train_stride: int = 40,
        eval_stride: int = 156,
        eps: float = 1e-8
    ) -> None:
        """
        初始化STFT切片数据集
        
        Args:
            raw_dir: 原始（含噪声）数据目录
            clean_dir: 干净数据目录
            mode: 数据集模式 ('train', 'val', 'test')
            window_size: 滑动窗口大小（帧数），默认156
            train_stride: 训练集滑动步长，默认40（高重叠扩充数据）
            eval_stride: 验证/测试集滑动步长，默认156（无重叠）
            eps: 数值稳定性常数，默认1e-8
        """
        super().__init__()
        
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.mode = mode.lower()
        self.window_size = window_size
        self.eps = eps
        
        # 根据模式选择步长
        self.stride = train_stride if self.mode == 'train' else eval_stride
        
        # 频域裁剪范围
        self.freq_start = self.FREQ_START
        self.freq_end = self.FREQ_END
        
        # 预计算所有切片索引
        self.slice_indices: List[Tuple[str, int]] = []
        self._precompute_slice_indices()
        
        print(f"[{self.mode.upper()}] 数据集初始化完成:")
        print(f"  - 文件数量: {len(self._get_file_list())}")
        print(f"  - 切片总数: {len(self.slice_indices)}")
        print(f"  - 窗口大小: {self.window_size}")
        print(f"  - 滑动步长: {self.stride}")
    
    def _get_file_list(self) -> List[str]:
        """
        获取数据目录中的所有.npy文件列表
        
        Returns:
            文件名列表（不含路径）
        """
        raw_files = set(f for f in os.listdir(self.raw_dir) if f.endswith('.npy'))
        clean_files = set(f for f in os.listdir(self.clean_dir) if f.endswith('.npy'))
        
        # 取交集，确保raw和clean都存在
        common_files = sorted(list(raw_files & clean_files))
        
        if len(common_files) == 0:
            raise ValueError(f"在 {self.raw_dir} 和 {self.clean_dir} 中未找到匹配的.npy文件")
        
        return common_files
    
    def _precompute_slice_indices(self) -> None:
        """
        预计算所有文件的切片索引
        
        遍历所有文件，根据时间维度长度和滑动窗口参数，
        计算每个有效切片的(文件名, 起始帧索引)元组。
        """
        file_list = self._get_file_list()
        
        for filename in file_list:
            # 加载文件获取时间维度长度
            raw_path = self.raw_dir / filename
            data = np.load(raw_path)
            
            # 预期形状: [2, 257, T_long]
            if data.ndim != 3 or data.shape[0] != 2:
                raise ValueError(f"文件 {filename} 形状异常: {data.shape}, 预期 [2, 257, T]")
            
            t_length = data.shape[2]
            
            # 计算可提取的切片数量
            if t_length < self.window_size:
                print(f"警告: 文件 {filename} 时间维度 {t_length} 小于窗口大小 {self.window_size}，跳过")
                continue
            
            # 滑动窗口切片
            num_slices = (t_length - self.window_size) // self.stride + 1
            
            for i in range(num_slices):
                start_idx = i * self.stride
                self.slice_indices.append((filename, start_idx))
    
    def _load_slice(
        self,
        filename: str,
        start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载指定文件的指定切片
        
        Args:
            filename: 文件名
            start_idx: 切片起始帧索引
            
        Returns:
            (raw_slice, clean_slice): 原始和干净数据的切片
        """
        # 加载原始数据
        raw_path = self.raw_dir / filename
        clean_path = self.clean_dir / filename
        
        raw_data = np.load(raw_path)
        clean_data = np.load(clean_path)
        
        # 提取时间窗口切片
        end_idx = start_idx + self.window_size
        raw_slice = raw_data[:, :, start_idx:end_idx]    # [2, 257, window_size]
        clean_slice = clean_data[:, :, start_idx:end_idx]
        
        # 频域裁剪: 保留频点索引 1 到 103
        raw_slice = raw_slice[:, self.freq_start:self.freq_end, :]    # [2, 103, 156]
        clean_slice = clean_slice[:, self.freq_start:self.freq_end, :]
        
        return raw_slice, clean_slice
    
    def _compute_magnitude(self, data: np.ndarray) -> np.ndarray:
        """
        计算复数STFT的幅度
        
        Args:
            data: 形状 [2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            幅度数组，形状 [F, T]
        """
        real = data[0]  # [F, T]
        imag = data[1]  # [F, T]
        magnitude = np.sqrt(real**2 + imag**2 + self.eps)
        return magnitude
    
    def _anti_identity_normalize(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        抗恒等映射归一化
        
        对数据进行Log变换后的Z-score归一化，
        并将归一化系数应用回复数实虚部以保持相位比例。
        
        Args:
            data: 形状 [2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            (normalized_data, mean, std): 归一化后的数据及统计量
        """
        # 1. 计算幅度
        magnitude = self._compute_magnitude(data)  # [F, T]
        
        # 2. Log变换
        log_magnitude = np.log1p(magnitude)  # log(1 + |S|)
        
        # 3. 计算统计量
        mean = float(np.mean(log_magnitude))
        std = float(np.std(log_magnitude) + self.eps)
        
        # 4. 计算归一化因子
        # 归一化后的log幅度: (log_mag - mean) / std
        # 原始幅度对应的缩放因子
        norm_factor = (log_magnitude - mean) / std  # [F, T]
        
        # 5. 将缩放因子应用到实虚部（保持相位）
        # 新的实虚部 = 原始实虚部 * (norm_factor / log_magnitude)
        # 这样保持相位不变，只改变幅度
        scale = norm_factor / (log_magnitude + self.eps)  # [F, T]
        
        normalized_data = np.zeros_like(data)
        normalized_data[0] = data[0] * scale  # Real
        normalized_data[1] = data[1] * scale  # Imag
        
        return normalized_data, mean, std
    
    def __len__(self) -> int:
        """返回数据集切片总数"""
        return len(self.slice_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的数据切片
        
        Args:
            idx: 切片索引
            
        Returns:
            包含以下键的字典:
                - 'input': 归一化后的原始数据 [2, 103, 156]
                - 'target': 归一化后的噪声残差 [2, 103, 156]
                - 'mean': 原始数据的归一化均值
                - 'std': 原始数据的归一化标准差
                - 'clean_norm': 归一化后的干净数据 [2, 103, 156]
        """
        filename, start_idx = self.slice_indices[idx]
        
        # 加载切片
        raw_slice, clean_slice = self._load_slice(filename, start_idx)
        
        # 抗恒等映射归一化
        raw_norm, raw_mean, raw_std = self._anti_identity_normalize(raw_slice)
        
        # 对clean使用相同的统计量进行归一化（保持一致性）
        # 计算clean的幅度和log变换
        clean_magnitude = self._compute_magnitude(clean_slice)
        clean_log_mag = np.log1p(clean_magnitude)
        
        # 使用raw的mean和std归一化clean
        clean_norm_factor = (clean_log_mag - raw_mean) / raw_std
        clean_scale = clean_norm_factor / (clean_log_mag + self.eps)
        
        clean_norm = np.zeros_like(clean_slice)
        clean_norm[0] = clean_slice[0] * clean_scale
        clean_norm[1] = clean_slice[1] * clean_scale
        
        # 计算残差目标: Noise = Raw - Clean
        noise_norm = raw_norm - clean_norm
        
        # 转换为PyTorch张量
        return {
            'input': torch.from_numpy(raw_norm).float(),      # [2, 103, 156]
            'target': torch.from_numpy(noise_norm).float(),   # [2, 103, 156]
            'mean': torch.tensor(raw_mean).float(),
            'std': torch.tensor(raw_std).float(),
            'clean_norm': torch.from_numpy(clean_norm).float()  # [2, 103, 156]
        }


def get_dataloaders(
    raw_base_dir: Union[str, Path],
    clean_base_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_stride: int = 40,
    eval_stride: int = 156,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取训练、验证和测试数据加载器
    
    Args:
        raw_base_dir: 原始数据基础目录（包含train/val/test子目录）
        clean_base_dir: 干净数据基础目录（包含train/val/test子目录）
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        train_stride: 训练集滑动步长
        eval_stride: 验证/测试集滑动步长
        pin_memory: 是否锁页内存
        
    Returns:
        (train_loader, val_loader, test_loader): 三个数据加载器
    """
    raw_base = Path(raw_base_dir)
    clean_base = Path(clean_base_dir)
    
    # 创建数据集
    train_dataset = STFTSlicingDataset(
        raw_dir=raw_base / 'train',
        clean_dir=clean_base / 'train',
        mode='train',
        train_stride=train_stride,
        eval_stride=eval_stride
    )
    
    val_dataset = STFTSlicingDataset(
        raw_dir=raw_base / 'val',
        clean_dir=clean_base / 'val',
        mode='val',
        train_stride=train_stride,
        eval_stride=eval_stride
    )
    
    test_dataset = STFTSlicingDataset(
        raw_dir=raw_base / 'test',
        clean_dir=clean_base / 'test',
        mode='test',
        train_stride=train_stride,
        eval_stride=eval_stride
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("STFTSlicingDataset 单元测试")
    print("=" * 60)
    
    # 创建模拟数据进行测试
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建目录结构
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(tmpdir, 'raw', split), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, 'clean', split), exist_ok=True)
        
        # 生成模拟数据
        for split in ['train', 'val', 'test']:
            for i in range(3):
                # 模拟STFT数据: [2, 257, T]
                t_length = 500 if split == 'train' else 300
                raw_data = np.random.randn(2, 257, t_length).astype(np.float32)
                clean_data = raw_data * 0.8 + np.random.randn(2, 257, t_length).astype(np.float32) * 0.1
                
                np.save(os.path.join(tmpdir, 'raw', split, f'sample_{i}.npy'), raw_data)
                np.save(os.path.join(tmpdir, 'clean', split, f'sample_{i}.npy'), clean_data)
        
        # 测试数据集
        train_loader, val_loader, test_loader = get_dataloaders(
            raw_base_dir=os.path.join(tmpdir, 'raw'),
            clean_base_dir=os.path.join(tmpdir, 'clean'),
            batch_size=4,
            num_workers=0
        )
        
        # 获取一个batch
        batch = next(iter(train_loader))
        
        print("\n批次数据形状:")
        print(f"  input:      {batch['input'].shape}")
        print(f"  target:     {batch['target'].shape}")
        print(f"  mean:       {batch['mean'].shape}")
        print(f"  std:        {batch['std'].shape}")
        print(f"  clean_norm: {batch['clean_norm'].shape}")
        
        print("\n✓ 数据集测试通过!")
