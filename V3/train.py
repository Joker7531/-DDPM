"""
Training Pipeline for Residual Noise Prediction U-Net (V3)

Key features:
1. Dual Loss: Noise prediction loss + Clean reconstruction loss
2. Log + InstanceNorm preprocessing in model
3. Residual learning paradigm

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
import time

from model import ResidualNoiseUNet
from dataset import ResidualSTFTDataset


class ResidualLoss(nn.Module):
    """
    Dual Loss for Residual Noise Prediction.
    
    Total Loss = alpha * NoiseLoss + beta * CleanLoss + gamma * LogMagLoss
    
    Where:
    - NoiseLoss: L1(predicted_noise, true_noise)
    - CleanLoss: L1(raw - predicted_noise, clean)
    - LogMagLoss: L1(log_mag(clean_pred), log_mag(clean))
    """
    
    def __init__(
        self,
        alpha: float = 1.0,  # Weight for noise prediction loss
        beta: float = 1.0,   # Weight for clean reconstruction loss
        gamma: float = 0.5,  # Weight for log-magnitude loss
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor,
        raw_stft: torch.Tensor,
        clean_stft: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred_noise: Predicted noise [B, 2, F, T]
            true_noise: True noise (raw - clean) [B, 2, F, T]
            raw_stft: Raw input STFT [B, 2, F, T]
            clean_stft: Ground truth clean STFT [B, 2, F, T]
            
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss values
        """
        # Noise prediction loss
        noise_loss = self.l1_loss(pred_noise, true_noise)
        
        # Clean reconstruction loss
        pred_clean = raw_stft - pred_noise
        clean_loss = self.l1_loss(pred_clean, clean_stft)
        
        # Log-magnitude loss for preserving spectral characteristics
        def log_magnitude(stft):
            real, imag = stft[:, 0], stft[:, 1]
            mag = torch.sqrt(real**2 + imag**2 + self.eps)
            return torch.log1p(mag)
        
        pred_log_mag = log_magnitude(pred_clean)
        true_log_mag = log_magnitude(clean_stft)
        log_mag_loss = self.l1_loss(pred_log_mag, true_log_mag)
        
        # Combined loss
        total_loss = self.alpha * noise_loss + self.beta * clean_loss + self.gamma * log_mag_loss
        
        loss_dict = {
            'noise_loss': noise_loss.item(),
            'clean_loss': clean_loss.item(),
            'log_mag_loss': log_mag_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """
    Trainer class for ResidualNoiseUNet.
    """
    
    def __init__(
        self,
        model: ResidualNoiseUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: ResidualLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: str = 'cuda',
        output_dir: str = './output_V3',
        grad_clip: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.grad_clip = grad_clip
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints_V3'
        self.log_dir = self.output_dir / 'logs_V3'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'noise_loss': [],
            'clean_loss': [],
            'log_mag_loss': [],
            'total_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (raw_stft, noise_stft) in enumerate(pbar):
            raw_stft = raw_stft.to(self.device)
            noise_stft = noise_stft.to(self.device)
            
            # Compute clean STFT for loss calculation
            clean_stft = raw_stft - noise_stft
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass: predict noise
            pred_noise = self.model(raw_stft)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                pred_noise, noise_stft, raw_stft, clean_stft
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'noise': f"{loss_dict['noise_loss']:.4f}",
                'clean': f"{loss_dict['clean_loss']:.4f}",
                'total': f"{loss_dict['total_loss']:.4f}"
            })
        
        # Average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        epoch_losses = {
            'noise_loss': [],
            'clean_loss': [],
            'log_mag_loss': [],
            'total_loss': []
        }
        
        pbar = tqdm(self.val_loader, desc="Validating")
        for raw_stft, noise_stft in pbar:
            raw_stft = raw_stft.to(self.device)
            noise_stft = noise_stft.to(self.device)
            clean_stft = raw_stft - noise_stft
            
            # Forward pass
            pred_noise = self.model(raw_stft)
            
            # Compute loss
            _, loss_dict = self.criterion(
                pred_noise, noise_stft, raw_stft, clean_stft
            )
            
            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"Starting Training (V3 - Residual Noise Prediction)")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"  Epochs: {num_epochs}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_losses = self.train_epoch()
            self.train_history.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            self.val_history.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n  Train - Noise: {train_losses['noise_loss']:.4f}, "
                  f"Clean: {train_losses['clean_loss']:.4f}, "
                  f"Total: {train_losses['total_loss']:.4f}")
            print(f"  Val   - Noise: {val_losses['noise_loss']:.4f}, "
                  f"Clean: {val_losses['clean_loss']:.4f}, "
                  f"Total: {val_losses['total_loss']:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                print(f"  ★ New best model! Val loss: {self.best_val_loss:.4f}")
            
            self._save_checkpoint(epoch, is_best)
            
            # Save training history
            self._save_history()
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': {
                'in_channels': self.model.in_channels,
                'out_channels': self.model.out_channels,
                'base_channels': self.model.base_channels,
                'depth': self.model.depth,
                'model_type': 'ResidualNoiseUNet'
            }
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch:03d}.pth')
    
    def _save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=4)


def main():
    """Main training script."""
    # Configuration
    config = {
        # Model
        'in_channels': 2,
        'out_channels': 2,
        'base_channels': 32,  # Reduced for memory efficiency
        'depth': 4,
        
        # Data
        'dataset_root': 'Dataset_STFT',
        'segment_length': 625,  # ~20s @ hop=32, fs=500
        'stride': 156,          # ~10s stride (50% overlap)
        
        # Training
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        
        # Loss weights
        'alpha': 1.0,  # Noise loss
        'beta': 1.0,   # Clean loss
        'gamma': 0.5,  # Log-mag loss
        
        # Output
        'output_dir': './output_V3',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4
    }
    
    print("=" * 70)
    print("V3: Residual Noise Prediction Training")
    print("=" * 70)
    
    # Create model
    model = ResidualNoiseUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        depth=config['depth']
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {n_params:,} ({n_params*4/1024/1024:.1f} MB)")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ResidualSTFTDataset(
        root_dir=config['dataset_root'],
        split='train',
        segment_length=config['segment_length'],
        stride=config['stride']
    )
    
    val_dataset = ResidualSTFTDataset(
        root_dir=config['dataset_root'],
        split='val',
        segment_length=config['segment_length'],
        stride=config['stride']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create loss function
    criterion = ResidualLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        output_dir=config['output_dir'],
        grad_clip=config['grad_clip']
    )
    
    # Save config
    config_path = Path(config['output_dir']) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
