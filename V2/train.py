"""
Training Script for SpectrogramUNet on EEG STFT Data

This module provides a complete training pipeline for the SpectrogramUNet model,
including training loop, validation, checkpointing, and logging.

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from model import SpectrogramUNet
from dataset import EEGSTFTDataset


class CompositeLoss(nn.Module):
    """
    Composite Loss for Complex STFT Spectrogram Reconstruction.
    
    This loss is specifically designed for EEG high-frequency reconstruction:
    
    Term 1: L1 Loss on Real/Imag parts
        - Direct reconstruction of real and imaginary components
    
    Term 2: Log-Magnitude L1 Loss
        - Magnitude: M = sqrt(Real^2 + Imag^2)
        - Loss: ||log(M + epsilon) - log(M_target + epsilon)||_1
        - Critical for preserving high-frequency content in EEG
    
    Args:
        l1_weight (float): Weight for L1 loss on real/imag (default: 1.0)
        log_mag_weight (float): Weight for log-magnitude loss (default: 1.0)
        epsilon (float): Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self, 
        l1_weight: float = 1.0, 
        log_mag_weight: float = 1.0,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.log_mag_weight = log_mag_weight
        self.epsilon = epsilon
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss for complex STFT spectrograms.
        
        Args:
            pred (torch.Tensor): Predicted STFT [B, 2, F, T]
                - pred[:, 0]: Real part
                - pred[:, 1]: Imaginary part
            target (torch.Tensor): Target STFT [B, 2, F, T]
                - target[:, 0]: Real part
                - target[:, 1]: Imaginary part
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Total loss and loss components
        """
        # Term 1: L1 Loss on Real/Imag parts
        l1_real_imag = self.l1_loss(pred, target)
        
        # Term 2: Log-Magnitude L1 Loss
        # Compute magnitude: M = sqrt(Real^2 + Imag^2)
        pred_magnitude = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + self.epsilon)
        target_magnitude = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + self.epsilon)
        
        # Log-magnitude loss
        log_mag_loss = self.l1_loss(
            torch.log(pred_magnitude + self.epsilon),
            torch.log(target_magnitude + self.epsilon)
        )
        
        # Combine losses
        total_loss = (
            self.l1_weight * l1_real_imag + 
            self.log_mag_weight * log_mag_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'l1_real_imag': l1_real_imag.item(),
            'log_magnitude': log_mag_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """
    Trainer class for SpectrogramUNet.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model (nn.Module): The SpectrogramUNet model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Training configuration
        device (str): Device to train on ('cuda' or 'cpu')
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory for TensorBoard logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        checkpoint_dir: str = './output_V2/checkpoints',
        log_dir: str = './output_V2/logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        
        # Setup loss, optimizer, scheduler
        self.criterion = CompositeLoss(
            l1_weight=config.get('l1_weight', 1.0),
            log_mag_weight=config.get('log_mag_weight', 1.0),
            epsilon=config.get('epsilon', 1e-8)
        )
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10)
        )
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(self.log_dir / f"run_{timestamp}")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Save config
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict[str, float]: Average training metrics
        """
        self.model.train()
        epoch_losses = {'total': [], 'mse': [], 'l1': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (raw_stft, clean_stft) in enumerate(pbar):
            # Move to device
            raw_stft = raw_stft.to(self.device)
            clean_stft = clean_stft.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_stft = self.model(raw_stft)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_stft, clean_stft)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Log
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'l1': f"{loss_dict['l1_real_imag']:.4f}",
                'log_mag': f"{loss_dict['log_magnitudese']:.4f}",
                'l1': f"{loss_dict['l1']:.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % self.config.get('log_interval', 10) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}_loss', value, self.global_step)
                
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict[str, float]: Average validation metrics
        """
        self.model.eval()
        epoch_losses = {'total': [], 'mse': [], 'l1': []}
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for raw_stft, clean_stft in pbar:
                # Move to device
                raw_stft = raw_stft.to(self.device)
                clean_stft = clean_stft.to(self.device)
                
                # Forward pass
                pred_stft = self.model(raw_stft)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_stft, clean_stft)
                
                # Log
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'l1': f"{loss_dict['l1_real_imag']:.4f}",
                    'log_mag': f"{loss_dict['log_magnitude']:.4f}"
                })
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # # Save periodic checkpoint
        # if (self.current_epoch + 1) % self.config.get('save_interval', 10) == 0:
        #     epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch + 1}.pth'
        #     torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs (int): Number of epochs to train
            resume_from (Optional[str]): Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Log to TensorBoard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f} (L1: {train_losses['l1_real_imag']:.4f}, LogMag: {train_losses['log_magnitude']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} (L1: {val_losses['l1_real_imag']:.4f}, LogMag: {val_losses['log_magnitude']:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            self.save_checkpoint(is_best)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 70)
        
        self.writer.close()


def main():
    """Main training script."""
    # Configuration
    config = {
        # Model
        'in_channels': 2,
        'out_channels': 2,
        'base_channels': 32,
        'depth': 4,
        
        # Dataset
        'dataset_root': 'Dataset',
        'fs': 500,
        'n_fft': 512,
        'hop_length': 64,
        
        # Training
        'batch_size': 64,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        
        # Loss
        'l1_weight': 1.0,
        'log_mag_weight': 1.0,
        'epsilon': 1e-8,
        
        # Scheduler
        'scheduler_patience': 10,
        
        # Logging
        'log_interval': 10,
        'save_interval': 10,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 6
    }
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = EEGSTFTDataset(
        root_dir=config['dataset_root'],
        split='train',
        fs=config['fs'],
        noverlap=config['n_fft'] - config['hop_length'],
        nfft=config['n_fft']
    )
    
    val_dataset = EEGSTFTDataset(
        root_dir=config['dataset_root'],
        split='val',
        fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        nfft=config['n_fft']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = SpectrogramUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        depth=config['depth']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device'],
        checkpoint_dir='./checkpoints_V2',
        log_dir='./logs_V2'
    )
    
    # Start training
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
