"""
训练脚本
包含训练循环、验证、模型保存和可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Optional
import json

from model import ConditionalDiffWave
from diffusion import GaussianDiffusion
from dataset import build_dataset_from_directory


def save_plot(
    raw: torch.Tensor,
    clean: torch.Tensor,
    denoised: torch.Tensor,
    epoch: int,
    save_dir: Path,
    prefix: str = 'viz'
) -> None:
    """
    可视化并保存对比图
    
    Args:
        raw: Raw输入 [1, 2048]
        clean: Ground truth [1, 2048]
        denoised: 模型预测 [1, 2048]
        epoch: 当前epoch
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    # 转换为numpy并去除通道维度
    raw_np = raw.squeeze().cpu().numpy()
    clean_np = clean.squeeze().cpu().numpy()
    denoised_np = denoised.squeeze().cpu().numpy()
    
    # 时间轴（秒）
    time_axis = np.arange(len(raw_np)) / 500  # 500Hz采样率
    
    # 创建三子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Top: Raw Input
    axes[0].plot(time_axis, raw_np, linewidth=0.8, color='#E74C3C', alpha=0.8)
    axes[0].set_title(f'Raw Input (Noisy) - Epoch {epoch}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim([time_axis[0], time_axis[-1]])
    
    # Middle: Clean Ground Truth
    axes[1].plot(time_axis, clean_np, linewidth=0.8, color='#27AE60', alpha=0.8)
    axes[1].set_title('Clean Ground Truth', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim([time_axis[0], time_axis[-1]])
    
    # Bottom: Model Prediction
    axes[2].plot(time_axis, denoised_np, linewidth=0.8, color='#3498DB', alpha=0.8)
    axes[2].set_title('Model Prediction (Denoised)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim([time_axis[0], time_axis[-1]])
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / f'{prefix}_epoch_{epoch:04d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved: {save_path.name}")


def train_epoch(
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        diffusion: 扩散模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        
    Returns:
        metrics: 平均损失字典
    """
    diffusion.train()
    
    total_loss = 0.0
    loss_accum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch_idx, (raw, clean) in enumerate(pbar):
        raw = raw.to(device)  # [B, 1, 2048]
        clean = clean.to(device)  # [B, 1, 2048]
        
        batch_size = raw.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        # 计算损失
        loss, loss_dict = diffusion.p_losses(clean, raw, t)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_accum[key] = loss_accum.get(key, 0.0) + value
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    # 计算平均
    num_batches = len(dataloader)
    metrics = {key: value / num_batches for key, value in loss_accum.items()}
    metrics['total'] = total_loss / num_batches
    
    return metrics


@torch.no_grad()
def validate_epoch(
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    viz_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    验证一个epoch
    
    Args:
        diffusion: 扩散模型
        dataloader: 验证数据加载器
        device: 设备
        epoch: 当前epoch
        viz_dir: 可视化保存目录（可选）
        
    Returns:
        metrics: 平均损失字典
    """
    diffusion.eval()
    
    total_loss = 0.0
    loss_accum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]', leave=False)
    
    # 采样标志（只采样第一个batch的第一个样本）
    sampled = False
    
    for batch_idx, (raw, clean) in enumerate(pbar):
        raw = raw.to(device)
        clean = clean.to(device)
        
        batch_size = raw.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        # 计算损失
        loss, loss_dict = diffusion.p_losses(clean, raw, t)
        
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_accum[key] = loss_accum.get(key, 0.0) + value
        
        pbar.set_postfix({'loss': loss.item()})
        
        # 可视化采样（只在第一个batch执行）
        if not sampled and viz_dir is not None:
            print(f"\n  Sampling for visualization...")
            # 采样第一个样本
            condition = raw[:1]  # [1, 1, 2048]
            denoised = diffusion.sample(condition)
            
            # 保存可视化
            save_plot(
                raw=raw[0],
                clean=clean[0],
                denoised=denoised[0],
                epoch=epoch,
                save_dir=viz_dir,
                prefix='val'
            )
            
            sampled = True
    
    # 计算平均
    num_batches = len(dataloader)
    metrics = {key: value / num_batches for key, value in loss_accum.items()}
    metrics['total'] = total_loss / num_batches
    
    return metrics


def train(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    device: str = 'cuda',
    resume: Optional[str] = None
):
    """
    主训练函数
    
    Args:
        data_dir: 数据根目录
        output_dir: 输出目录（保存模型和可视化）
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        num_workers: 数据加载线程数
        device: 训练设备
        resume: 恢复训练的检查点路径（可选）
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    viz_dir = output_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    log_path = output_path / 'training_log.json'
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 构建数据集
    print("\n=== Loading Datasets ===")
    train_dataset = build_dataset_from_directory(
        data_dir=data_dir,
        split='train',
        segment_length=2048,
        augment=True
    )
    
    val_dataset = build_dataset_from_directory(
        data_dir=data_dir,
        split='val',
        segment_length=2048,
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 创建模型
    print("\n=== Building Model ===")
    model = ConditionalDiffWave(
        in_channels=2,
        out_channels=1,
        residual_channels=256,
        num_layers=30,
        dilation_cycle=10,
        time_emb_dim=512,
        condition_dropout=0.1  # 条件Dropout提高鲁棒性
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        loss_type='hybrid'
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    training_log = []
    
    if resume is not None:
        print(f"\n=== Resuming from {resume} ===")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                training_log = json.load(f)
        
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
    
    # 训练循环
    print("\n=== Starting Training ===")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {output_path}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_metrics = train_epoch(diffusion, train_loader, optimizer, device, epoch+1)
        
        print(f"\n[Train] Loss: {train_metrics['total']:.6f}")
        if 'noise_l1' in train_metrics:
            print(f"  Noise L1: {train_metrics['noise_l1']:.6f}")
        if 'stft_total' in train_metrics:
            print(f"  STFT: {train_metrics['stft_total']:.6f}")
        
        # 验证
        val_metrics = validate_epoch(diffusion, val_loader, device, epoch+1, viz_dir)
        
        print(f"\n[Val] Loss: {val_metrics['total']:.6f}")
        if 'noise_l1' in val_metrics:
            print(f"  Noise L1: {val_metrics['noise_l1']:.6f}")
        if 'stft_total' in val_metrics:
            print(f"  STFT: {val_metrics['stft_total']:.6f}")
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(f"\nLearning rate: {current_lr:.2e}")
        
        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr
        }
        training_log.append(log_entry)
        
        # 保存日志
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # 保存最新模型（覆盖式）
        latest_path = checkpoint_dir / 'model_latest.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['total'],
            'val_loss': val_metrics['total'],
            'best_val_loss': best_val_loss
        }, latest_path)
        
        # 保存最佳模型
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_path = checkpoint_dir / 'model_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total'],
                'val_loss': val_metrics['total'],
                'best_val_loss': best_val_loss
            }, best_path)
            print(f"\n✓ New best model saved! Val Loss: {best_val_loss:.6f}")
        
        # 可选：删除旧的可视化（只保留best和latest对应的）
        # 这里保留所有以观察训练过程
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved in: {checkpoint_dir}")
    print(f"Visualizations saved in: {viz_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Conditional DDPM for EEG Denoising')
    
    parser.add_argument('--data_dir', type=str, default='Dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
