"""
训练脚本（混合精度优化版）
针对RTX 4090D优化，支持AMP混合精度训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Optional
import json
import time

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
    """可视化并保存对比图"""
    raw_np = raw.squeeze().cpu().numpy()
    clean_np = clean.squeeze().cpu().numpy()
    denoised_np = denoised.squeeze().cpu().numpy()
    
    time_axis = np.arange(len(raw_np)) / 500
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(time_axis, raw_np, linewidth=0.8, color='#E74C3C', alpha=0.8)
    axes[0].set_title(f'Raw Input (Noisy) - Epoch {epoch}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim([time_axis[0], time_axis[-1]])
    
    axes[1].plot(time_axis, clean_np, linewidth=0.8, color='#27AE60', alpha=0.8)
    axes[1].set_title('Clean Ground Truth', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim([time_axis[0], time_axis[-1]])
    
    axes[2].plot(time_axis, denoised_np, linewidth=0.8, color='#3498DB', alpha=0.8)
    axes[2].set_title('Model Prediction (Denoised)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim([time_axis[0], time_axis[-1]])
    
    plt.tight_layout()
    
    save_path = save_dir / f'{prefix}_epoch_{epoch:04d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved: {save_path.name}")


def train_epoch(
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> Dict[str, float]:
    """训练一个epoch（支持混合精度）"""
    diffusion.train()
    
    total_loss = 0.0
    loss_accum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch_idx, (raw, clean) in enumerate(pbar):
        raw = raw.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        
        batch_size = raw.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        # 混合精度训练
        if use_amp:
            with autocast():
                loss, loss_dict = diffusion.p_losses(clean, raw, t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = diffusion.p_losses(clean, raw, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_accum[key] = loss_accum.get(key, 0.0) + value
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
    viz_dir: Optional[Path] = None,
    use_amp: bool = True
) -> Dict[str, float]:
    """验证一个epoch"""
    diffusion.eval()
    
    total_loss = 0.0
    loss_accum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]', leave=False)
    sampled = False
    
    for batch_idx, (raw, clean) in enumerate(pbar):
        raw = raw.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        
        batch_size = raw.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        # 验证时也使用混合精度以加速
        if use_amp:
            with autocast():
                loss, loss_dict = diffusion.p_losses(clean, raw, t)
        else:
            loss, loss_dict = diffusion.p_losses(clean, raw, t)
        
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_accum[key] = loss_accum.get(key, 0.0) + value
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 可视化采样
        if not sampled and viz_dir is not None:
            print(f"\n  Sampling for visualization...")
            condition = raw[:1]
            
            if use_amp:
                with autocast():
                    denoised = diffusion.sample(condition)
            else:
                denoised = diffusion.sample(condition)
            
            save_plot(
                raw=raw[0],
                clean=clean[0],
                denoised=denoised[0],
                epoch=epoch,
                save_dir=viz_dir,
                prefix='val'
            )
            
            sampled = True
    
    num_batches = len(dataloader)
    metrics = {key: value / num_batches for key, value in loss_accum.items()}
    metrics['total'] = total_loss / num_batches
    
    return metrics


def train(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 300,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_workers: int = 8,
    device: str = 'cuda',
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    resume: Optional[str] = None
):
    """主训练函数（优化版）"""
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"Gradient Accumulation Steps: {grad_accum_steps}")
    
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
    
    # 创建数据加载器（优化设置）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 创建模型
    print("\n=== Building Model ===")
    model = ConditionalDiffWave(
        in_channels=2,
        out_channels=1,
        residual_channels=256,
        num_layers=30,
        dilation_cycle=10,
        time_emb_dim=512
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
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 混合精度训练的Scaler
    scaler = GradScaler(enabled=use_amp)
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    training_log = []
    
    if resume is not None:
        print(f"\n=== Resuming from {resume} ===")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                training_log = json.load(f)
        
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
    
    # 训练循环
    print("\n=== Starting Training ===")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {output_path}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_metrics = train_epoch(diffusion, train_loader, optimizer, scaler, device, epoch+1, use_amp)
        
        print(f"\n[Train] Loss: {train_metrics['total']:.6f}")
        if 'noise_l1' in train_metrics:
            print(f"  Noise L1: {train_metrics['noise_l1']:.6f}")
        if 'stft_total' in train_metrics:
            print(f"  STFT: {train_metrics['stft_total']:.6f}")
        
        # 验证
        val_metrics = validate_epoch(diffusion, val_loader, device, epoch+1, viz_dir, use_amp)
        
        print(f"\n[Val] Loss: {val_metrics['total']:.6f}")
        if 'noise_l1' in val_metrics:
            print(f"  Noise L1: {val_metrics['noise_l1']:.6f}")
        if 'stft_total' in val_metrics:
            print(f"  STFT: {val_metrics['stft_total']:.6f}")
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"\nLearning rate: {current_lr:.2e}")
        print(f"Epoch time: {epoch_time:.1f}s")
        
        # 显存使用情况
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'epoch_time': epoch_time
        }
        training_log.append(log_entry)
        
        # 保存日志
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # 保存最新模型
        latest_path = checkpoint_dir / 'model_latest.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
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
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_metrics['total'],
                'val_loss': val_metrics['total'],
                'best_val_loss': best_val_loss
            }, best_path)
            print(f"\n✓ New best model saved! Val Loss: {best_val_loss:.6f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved in: {checkpoint_dir}")
    print(f"Visualizations saved in: {viz_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Conditional DDPM for EEG Denoising (Optimized for RTX 4090D)')
    
    parser.add_argument('--data_dir', type=str, default='Dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (推荐: 16-32 for RTX 4090D)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable AMP')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
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
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
