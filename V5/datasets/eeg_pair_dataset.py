"""
EEG Pair Dataset for Single-Channel Artifact Removal
支持 raw-clean 配对数据集，并提供灵活的切片、归一化策略
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Literal, Dict, Tuple


class EEGPairDataset(Dataset):
    """
    单通道 EEG raw-clean 配对数据集
    
    Args:
        root: 数据集根目录，包含 train/val/test 子目录
        split: "train", "val", "test"
        segment_length: 切片长度（None 表示使用全长）
        random_crop: 是否随机裁剪（仅 train 推荐 True）
        stride: 滑窗步长（仅当 random_crop=False 且 segment_length 指定时生效）
        normalize: "none" 或 "zscore_per_sample"
        return_meta: 是否返回元数据（文件名、起始点等）
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        segment_length: Optional[int] = None,
        random_crop: bool = True,
        stride: Optional[int] = None,
        normalize: Literal["none", "zscore_per_sample"] = "zscore_per_sample",
        return_meta: bool = False,
        transform=None,  # 数据增强
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.segment_length = segment_length
        self.random_crop = random_crop
        self.stride = stride if stride is not None else (segment_length if segment_length else 0)
        self.normalize = normalize
        self.return_meta = return_meta
        self.transform = transform  # 数据增强（仅对训练集应用）
        
        # 构建文件路径
        self.raw_dir = self.root / split / "raw"
        self.clean_dir = self.root / split / "clean"
        
        assert self.raw_dir.exists(), f"Raw directory not found: {self.raw_dir}"
        assert self.clean_dir.exists(), f"Clean directory not found: {self.clean_dir}"
        
        # 扫描配对文件
        self.file_pairs = self._scan_pairs()
        
        # 如果指定了 segment_length 且不使用随机裁剪，预计算所有切片索引
        if self.segment_length and not self.random_crop:
            self.slice_indices = self._precompute_slices()
        else:
            self.slice_indices = None
    
    def _scan_pairs(self) -> list:
        """扫描 raw-clean 配对文件"""
        raw_files = sorted([f.name for f in self.raw_dir.glob("*.npy")])
        clean_files = sorted([f.name for f in self.clean_dir.glob("*.npy")])
        
        # 提取基础名称（去除 _raw 或 _clean 后缀）
        def get_base_name(filename: str) -> str:
            """从文件名中提取基础ID
            支持格式：
            - train_001_raw.npy -> train_001
            - sub01_raw.npy -> sub01
            - 001.npy -> 001
            """
            name = filename.replace(".npy", "")
            # 移除常见后缀
            for suffix in ["_raw", "_clean", "_noisy", "_denoised"]:
                if name.endswith(suffix):
                    return name[:-len(suffix)]
            return name
        
        # 建立基础名到文件名的映射
        raw_map = {get_base_name(f): f for f in raw_files}
        clean_map = {get_base_name(f): f for f in clean_files}
        
        # 找到共同的基础名
        common_bases = set(raw_map.keys()) & set(clean_map.keys())
        
        if len(common_bases) == 0:
            error_msg = f"No matching pairs found in {self.split}\n"
            error_msg += f"  Raw dir: {self.raw_dir} ({len(raw_files)} files)\n"
            error_msg += f"  Clean dir: {self.clean_dir} ({len(clean_files)} files)\n"
            error_msg += f"  Please check:\n"
            error_msg += f"    1. Dataset root path is correct\n"
            error_msg += f"    2. Files exist in both raw/ and clean/ directories\n"
            error_msg += f"    3. File names match between raw/ and clean/ (after removing _raw/_clean suffixes)\n"
            if len(raw_files) > 0:
                error_msg += f"  Example raw files: {raw_files[:3]}\n"
            if len(clean_files) > 0:
                error_msg += f"  Example clean files: {clean_files[:3]}\n"
            if len(raw_files) > 0 and len(clean_files) > 0:
                error_msg += f"  Example base names:\n"
                error_msg += f"    Raw: {[get_base_name(f) for f in raw_files[:3]]}\n"
                error_msg += f"    Clean: {[get_base_name(f) for f in clean_files[:3]]}\n"
            raise AssertionError(error_msg)
        
        if len(common_bases) < len(raw_map):
            print(f"Warning: {len(raw_map) - len(common_bases)} raw files missing clean counterpart")
        if len(common_bases) < len(clean_map):
            print(f"Warning: {len(clean_map) - len(common_bases)} clean files missing raw counterpart")
        
        # 返回配对：[(raw_filename, clean_filename), ...]
        pairs = [(raw_map[base], clean_map[base]) for base in sorted(common_bases)]
        print(f"[{self.split}] Found {len(pairs)} valid pairs")
        return pairs
    
    def _precompute_slices(self) -> list:
        """
        预计算所有样本的滑窗切片索引
        返回: [(file_idx, start_idx), ...]
        """
        slices = []
        for file_idx, (raw_fname, clean_fname) in enumerate(self.file_pairs):
            raw_path = self.raw_dir / raw_fname
            arr = np.load(raw_path)
            L = arr.shape[-1] if arr.ndim == 2 else len(arr)
            
            if L < self.segment_length:
                # 长度不足，仅一个切片（后续会 zero-pad）
                slices.append((file_idx, 0))
            else:
                # 滑窗
                num_slices = (L - self.segment_length) // self.stride + 1
                for i in range(num_slices):
                    start = i * self.stride
                    slices.append((file_idx, start))
        
        print(f"[{self.split}] Precomputed {len(slices)} slices from {len(self.file_pairs)} files")
        return slices
    
    def __len__(self) -> int:
        if self.slice_indices is not None:
            return len(self.slice_indices)
        else:
            return len(self.file_pairs)
    
    def _load_and_unify(self, path: Path) -> np.ndarray:
        """
        加载 .npy 并统一为 shape (1, L)
        """
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[None, :]  # (L,) -> (1, L)
        elif arr.ndim == 2:
            if arr.shape[0] != 1:
                # 如果是 (L, 1)，转置
                if arr.shape[1] == 1:
                    arr = arr.T
                else:
                    raise ValueError(f"Unexpected shape {arr.shape} in {path}")
        else:
            raise ValueError(f"Unexpected ndim {arr.ndim} in {path}")
        
        assert arr.shape[0] == 1, f"Expected single channel, got {arr.shape}"
        return arr.astype(np.float32)
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization per sample"""
        if self.normalize == "zscore_per_sample":
            mean = x.mean()
            std = x.std()
            if std < 1e-8:
                std = 1.0
            x = (x - mean) / std
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        返回:
            x_raw: (1, L_seg) float32
            x_clean: (1, L_seg) float32
            meta: dict
        """
        # 确定文件索引和起始点
        if self.slice_indices is not None:
            file_idx, start = self.slice_indices[idx]
        else:
            file_idx = idx
            start = None  # 稍后根据策略确定
        
        raw_fname, clean_fname = self.file_pairs[file_idx]
        raw_path = self.raw_dir / raw_fname
        clean_path = self.clean_dir / clean_fname
        
        # 加载数据
        x_raw = self._load_and_unify(raw_path)    # (1, L)
        x_clean = self._load_and_unify(clean_path)
        
        L = x_raw.shape[1]
        assert x_clean.shape[1] == L, "Raw and clean length mismatch"
        
        # 切片策略
        if self.segment_length is None:
            # 使用全长
            seg_raw = x_raw
            seg_clean = x_clean
            actual_start = 0
            is_padded = False
        else:
            if L < self.segment_length:
                # 零填充
                pad_len = self.segment_length - L
                seg_raw = np.pad(x_raw, ((0, 0), (0, pad_len)), mode='constant')
                seg_clean = np.pad(x_clean, ((0, 0), (0, pad_len)), mode='constant')
                actual_start = 0
                is_padded = True
            else:
                # 裁剪
                if start is not None:
                    # 已预计算的起始点
                    actual_start = start
                elif self.random_crop:
                    # 随机裁剪
                    max_start = L - self.segment_length
                    actual_start = np.random.randint(0, max_start + 1)
                else:
                    # 中心裁剪
                    actual_start = (L - self.segment_length) // 2
                
                seg_raw = x_raw[:, actual_start:actual_start + self.segment_length]
                seg_clean = x_clean[:, actual_start:actual_start + self.segment_length]
                is_padded = False
        
        # 归一化：使用 raw 的统计量同时归一化两者（保持加性噪声假设）
        if self.normalize == "zscore_per_sample":
            # 仅计算 raw 的统计量
            mean_raw = seg_raw.mean()
            std_raw = seg_raw.std()
            if std_raw < 1e-8:
                std_raw = 1.0
            # 用相同的统计量归一化两者
            seg_raw = (seg_raw - mean_raw) / std_raw
            seg_clean = (seg_clean - mean_raw) / std_raw
        
        # 转 tensor
        x_raw_t = torch.from_numpy(seg_raw).float()
        x_clean_t = torch.from_numpy(seg_clean).float()
        
        # 应用数据增强（仅训练集）
        if self.transform is not None and self.split == "train":
            x_raw_t, x_clean_t = self.transform(x_raw_t, x_clean_t)
        
        # 构建 meta
        meta = {
            "raw_filename": raw_fname,
            "clean_filename": clean_fname,
            "start": actual_start,
            "original_length": L,
            "is_padded": is_padded,
        }
        
        # Shape assertions
        if self.segment_length:
            assert x_raw_t.shape == (1, self.segment_length), f"Expected (1, {self.segment_length}), got {x_raw_t.shape}"
            assert x_clean_t.shape == (1, self.segment_length)
        
        if self.return_meta:
            return x_raw_t, x_clean_t, meta
        else:
            return x_raw_t, x_clean_t


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eeg_pair_dataset.py <dataset_root>")
        sys.exit(0)
    
    root = sys.argv[1]
    
    # 测试 train
    ds_train = EEGPairDataset(
        root=root,
        split="train",
        segment_length=2048,
        random_crop=True,
        normalize="zscore_per_sample",
        return_meta=True
    )
    print(f"Train dataset length: {len(ds_train)}")
    
    x_raw, x_clean, meta = ds_train[0]
    print(f"Sample shape: {x_raw.shape}, {x_clean.shape}")
    print(f"Meta: {meta}")
    print(f"Raw range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
    
    # 测试 val with sliding window
    ds_val = EEGPairDataset(
        root=root,
        split="val",
        segment_length=2048,
        random_crop=False,
        stride=1024,
        normalize="zscore_per_sample",
        return_meta=True
    )
    print(f"\nVal dataset length: {len(ds_val)}")
    x_raw, x_clean, meta = ds_val[0]
    print(f"Sample shape: {x_raw.shape}")
    print(f"Meta: {meta}")
