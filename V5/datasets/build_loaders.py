"""
构建 train/val/test DataLoaders
"""
from torch.utils.data import DataLoader
from typing import Dict, Optional
from .eeg_pair_dataset import EEGPairDataset


def build_dataloaders(
    root: str,
    batch_size: int = 16,
    segment_length: Optional[int] = 2048,
    train_stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    normalize: str = "zscore_per_sample",
    num_workers: int = 4,
    pin_memory: bool = True,
    return_meta: bool = False,
    train_transform=None,  # 训练集数据增强
) -> Dict[str, DataLoader]:
    """
    构建 train/val/test DataLoaders
    
    Args:
        root: 数据集根目录
        batch_size: batch size
        segment_length: 切片长度（None 表示全长）
        train_stride: train 滑窗步长（不使用滑窗时设为 None）
        val_stride: val 滑窗步长（推荐设置为 segment_length//2）
        test_stride: test 滑窗步长（推荐设置为 segment_length//2）
        normalize: 归一化方式
        num_workers: DataLoader workers
        pin_memory: 是否 pin memory
        return_meta: 是否返回 meta 信息
    
    Returns:
        {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    
    # Train dataset: 使用滑窗切片（如果提供了stride）或随机裁剪
    train_ds = EEGPairDataset(
        root=root,
        split="train",
        segment_length=segment_length,
        random_crop=(train_stride is None),  # 仅在未提供stride时随机裁剪
        stride=train_stride if train_stride else segment_length,
        normalize=normalize,
        return_meta=return_meta,
        transform=train_transform,  # 数据增强
    )
    
    # Val dataset: 确定性切片（滑窗或中心裁剪）
    val_ds = EEGPairDataset(
        root=root,
        split="val",
        segment_length=segment_length,
        random_crop=False,
        stride=val_stride if val_stride else segment_length,  # 默认无重叠
        normalize=normalize,
        return_meta=return_meta,
    )
    
    # Test dataset: 确定性切片
    test_ds = EEGPairDataset(
        root=root,
        split="test",
        segment_length=segment_length,
        random_crop=False,
        stride=test_stride if test_stride else segment_length,
        normalize=normalize,
        return_meta=return_meta,
    )
    
    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 训练时丢弃最后不完整 batch
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    print(f"✓ Built DataLoaders:")
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_loaders.py <dataset_root>")
        sys.exit(0)
    
    root = sys.argv[1]
    loaders = build_dataloaders(
        root=root,
        batch_size=8,
        segment_length=2048,
        val_stride=1024,
        test_stride=1024,
        num_workers=0,
        return_meta=True,
    )
    
    # 测试一个 batch
    batch = next(iter(loaders["train"]))
    if len(batch) == 3:
        x_raw, x_clean, meta = batch
        print(f"\nTrain batch shapes: {x_raw.shape}, {x_clean.shape}")
        print(f"Meta keys: {meta.keys()}")
    else:
        x_raw, x_clean = batch
        print(f"\nTrain batch shapes: {x_raw.shape}, {x_clean.shape}")
