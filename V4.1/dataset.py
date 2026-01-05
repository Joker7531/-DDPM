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
        eps: float = 1e-6
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
        
        # 预计算所有切片索引: (raw_file, clean_file, start_idx)
        self.slice_indices: List[Tuple[str, str, int]] = []
        self._precompute_slice_indices()
        
        print(f"[{self.mode.upper()}] 数据集初始化完成:")
        print(f"  - 文件数量: {len(self._get_file_list())}")
        print(f"  - 切片总数: {len(self.slice_indices)}")
        print(f"  - 窗口大小: {self.window_size}")
        print(f"  - 滑动步长: {self.stride}")
    
    def _get_file_list(self) -> List[Tuple[str, str]]:
        """
        获取数据目录中的所有匹配的.npy文件对列表
        
        文件名格式:
            - raw目录: {split}_{id}_raw.npy (如 train_001_raw.npy)
            - clean目录: {split}_{id}_clean.npy (如 train_001_clean.npy)
        
        Returns:
            文件对列表: [(raw_filename, clean_filename), ...]
        """
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith('_raw.npy')]
        clean_files = set(f for f in os.listdir(self.clean_dir) if f.endswith('_clean.npy'))
        
        # 匹配raw和clean文件
        # raw: train_001_raw.npy -> 提取 train_001
        # clean: train_001_clean.npy
        matched_pairs = []
        for raw_file in raw_files:
            # 提取基础名称: train_001_raw.npy -> train_001
            base_name = raw_file.replace('_raw.npy', '')
            clean_file = f"{base_name}_clean.npy"
            
            if clean_file in clean_files:
                matched_pairs.append((raw_file, clean_file))
        
        matched_pairs = sorted(matched_pairs, key=lambda x: x[0])
        
        if len(matched_pairs) == 0:
            raise ValueError(
                f"在 {self.raw_dir} 和 {self.clean_dir} 中未找到匹配的文件对\n"
                f"预期格式: raw目录下 *_raw.npy, clean目录下 *_clean.npy"
            )
        
        return matched_pairs
    
    def _precompute_slice_indices(self) -> None:
        """
        预计算所有文件的切片索引
        
        遍历所有文件，根据时间维度长度和滑动窗口参数，
        计算每个有效切片的(raw文件名, clean文件名, 起始帧索引)元组。
        """
        file_pairs = self._get_file_list()
        
        for raw_file, clean_file in file_pairs:
            # 加载文件获取时间维度长度
            raw_path = self.raw_dir / raw_file
            data = np.load(raw_path)
            
            # 预期形状: [2, 257, T_long]
            if data.ndim != 3 or data.shape[0] != 2:
                raise ValueError(f"文件 {raw_file} 形状异常: {data.shape}, 预期 [2, 257, T]")
            
            t_length = data.shape[2]
            
            # 计算可提取的切片数量
            if t_length < self.window_size:
                print(f"警告: 文件 {raw_file} 时间维度 {t_length} 小于窗口大小 {self.window_size}，跳过")
                continue
            
            # 滑动窗口切片
            num_slices = (t_length - self.window_size) // self.stride + 1
            
            for i in range(num_slices):
                start_idx = i * self.stride
                self.slice_indices.append((raw_file, clean_file, start_idx))
    
    def _load_slice(
        self,
        raw_file: str,
        clean_file: str,
        start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载指定文件的指定切片
        
        Args:
            raw_file: raw数据文件名 (如 train_001_raw.npy)
            clean_file: clean数据文件名 (如 train_001_clean.npy)
            start_idx: 切片起始帧索引
            
        Returns:
            (raw_slice, clean_slice): 原始和干净数据的切片
        """
        # 加载原始数据
        raw_path = self.raw_dir / raw_file
        clean_path = self.clean_dir / clean_file
        
        raw_data = np.load(raw_path)
        clean_data = np.load(clean_path)
        
        # 提取时间窗口切片
        end_idx = start_idx + self.window_size
        raw_slice = raw_data[:, :, start_idx:end_idx]    # [2, 257, window_size]
        clean_slice = clean_data[:, :, start_idx:end_idx]
        
        # 频域裁剪: 保留频点索引 1 到 103
        raw_slice = raw_slice[:, self.freq_start:self.freq_end, :]    # [2, 103, 156]
        clean_slice = clean_slice[:, self.freq_start:self.freq_end, :]
        
        # 数据验证和清洗
        # 处理 NaN 和 Inf
        if not np.isfinite(raw_slice).all():
            raw_slice = np.nan_to_num(raw_slice, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(clean_slice).all():
            clean_slice = np.nan_to_num(clean_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 限制极端值 (防止数值溢出)
        max_val = 1e6
        raw_slice = np.clip(raw_slice, -max_val, max_val)
        clean_slice = np.clip(clean_slice, -max_val, max_val)
        
        return raw_slice, clean_slice
    
    def _instance_normalize(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Instance Normalization: 对单个切片直接进行 Z-score 归一化
        
        直接对 Real/Imag 通道进行操作，不做幅度-相位分离，不做 Log 变换。
        输入端保持线性缩放，Log 压缩放到 Loss Function 中处理。
        
        Args:
            data: 形状 [2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            (normalized_data, mean, std):
                - normalized_data: 归一化后的数据 [2, F, T]
                - mean: 该切片的均值 (标量)
                - std: 该切片的标准差 (标量)
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
    
    def __len__(self) -> int:
        """返回数据集切片总数"""
        return len(self.slice_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的数据切片
        
        使用 Instance Normalization：每个切片独立计算 mean/std，
        直接对 Real/Imag 进行线性缩放，不做 Log 变换。
        
        Args:
            idx: 切片索引
            
        Returns:
            包含以下键的字典:
                - 'input': 归一化后的原始数据 [2, 103, 156]
                - 'target': 归一化后的干净数据 [2, 103, 156] (直接预测clean，非残差)
                - 'mean': raw切片的归一化均值 (用于反归一化)
                - 'std': raw切片的归一化标准差 (用于反归一化)
                - 'raw_slice': 原始未归一化的raw数据 [2, 103, 156] (用于Loss计算)
                - 'clean_slice': 原始未归一化的clean数据 [2, 103, 156] (用于Loss计算)
        """
        raw_file, clean_file, start_idx = self.slice_indices[idx]
        
        # 加载切片 (已完成频域裁剪和数据清洗)
        raw_slice, clean_slice = self._load_slice(raw_file, clean_file, start_idx)
        
        # Instance Normalization: 使用 raw 切片的统计量
        raw_norm, raw_mean, raw_std = self._instance_normalize(raw_slice)
        
        # 对 clean 使用相同的统计量进行归一化 (保持一致性)
        clean_norm = (clean_slice - raw_mean) / raw_std
        clean_norm = np.clip(clean_norm, -10.0, 10.0)
        if not np.isfinite(clean_norm).all():
            clean_norm = np.nan_to_num(clean_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算残差目标: Noise = Raw - Clean (在归一化域)
        noise_norm = raw_norm - clean_norm
        if not np.isfinite(noise_norm).all():
            noise_norm = np.nan_to_num(noise_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为 PyTorch 张量
        input_tensor = torch.from_numpy(raw_norm.copy()).float()
        target_tensor = torch.from_numpy(noise_norm.copy()).float()  # 残差目标
        clean_norm_tensor = torch.from_numpy(clean_norm.copy()).float()
        raw_tensor = torch.from_numpy(raw_slice.copy()).float()
        clean_tensor = torch.from_numpy(clean_slice.copy()).float()
        
        # 安全检查
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        clean_norm_tensor = torch.nan_to_num(clean_norm_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'input': input_tensor,            # [2, 103, 156] 归一化后的 raw
            'target': target_tensor,          # [2, 103, 156] 归一化后的噪声残差 (raw - clean)
            'clean_norm': clean_norm_tensor,  # [2, 103, 156] 归一化后的 clean
            'mean': torch.tensor(raw_mean).float(),
            'std': torch.tensor(raw_std).float(),
            'raw_slice': raw_tensor,          # [2, 103, 156] 原始 raw (用于 Loss)
            'clean_slice': clean_tensor       # [2, 103, 156] 原始 clean (用于 Loss)
        }


def get_dataloaders(
    data_base_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_stride: int = 40,
    eval_stride: int = 156,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取训练、验证和测试数据加载器
    
    支持两种目录结构：
    
    结构1（推荐）:
        data_base_dir/
        ├── train/
        │   ├── raw/        # npy文件
        │   └── clean/      # npy文件
        ├── val/
        │   ├── raw/
        │   └── clean/
        └── test/
            ├── raw/
            └── clean/
    
    Args:
        data_base_dir: 数据基础目录（包含train/val/test子目录，每个下有raw/clean）
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        train_stride: 训练集滑动步长
        eval_stride: 验证/测试集滑动步长
        pin_memory: 是否锁页内存
        
    Returns:
        (train_loader, val_loader, test_loader): 三个数据加载器
    """
    data_base = Path(data_base_dir)
    
    # 创建数据集
    train_dataset = STFTSlicingDataset(
        raw_dir=data_base / 'train' / 'raw',
        clean_dir=data_base / 'train' / 'clean',
        mode='train',
        train_stride=train_stride,
        eval_stride=eval_stride
    )
    
    val_dataset = STFTSlicingDataset(
        raw_dir=data_base / 'val' / 'raw',
        clean_dir=data_base / 'val' / 'clean',
        mode='val',
        train_stride=train_stride,
        eval_stride=eval_stride
    )
    
    test_dataset = STFTSlicingDataset(
        raw_dir=data_base / 'test' / 'raw',
        clean_dir=data_base / 'test' / 'clean',
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
