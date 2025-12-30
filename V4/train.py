"""
SpectrogramNAFNet 训练脚本

本模块实现完整的训练流程，包括:
- 数据加载
- 模型初始化
- 优化器和学习率调度器配置
- 训练循环
- 验证评估
- 模型保存和检查点管理

作者: AI Assistant
日期: 2025-12-30
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from dataset import STFTSlicingDataset, get_dataloaders
from model import SpectrogramNAFNet, count_parameters
from losses import CompositeLoss, PSNRMetric


# 配置日志
def setup_logger(log_dir: Path, name: str = 'train') -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        log_dir: 日志目录
        name: 日志记录器名称
        
    Returns:
        配置好的日志记录器
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'{name}_{timestamp}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AverageMeter:
    """
    计算和存储平均值和当前值
    
    用于跟踪训练过程中的损失和指标。
    """
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计量"""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计量
        
        Args:
            val: 当前值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class Trainer:
    """
    SpectrogramNAFNet 训练器
    
    封装完整的训练流程，包括训练、验证和模型保存。
    
    Args:
        model: SpectrogramNAFNet 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        config: 配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: Dict[str, Any]
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.logger = setup_logger(self.output_dir / 'logs')
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # PSNR 指标
        self.psnr_metric = PSNRMetric()
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        self.best_psnr = float('-inf')
        
        # 当前 epoch
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        执行一个训练 epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        
        # 损失统计
        loss_meter = AverageMeter()
        noise_loss_meter = AverageMeter()
        reconstruct_loss_meter = AverageMeter()
        
        # 进度统计
        start_time = time.time()
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            raw_norm = batch['input'].to(self.device)       # [B, 2, 103, 156]
            noise_target = batch['target'].to(self.device)  # [B, 2, 103, 156]
            clean_norm = batch['clean_norm'].to(self.device)  # [B, 2, 103, 156]
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播 (混合精度)
            if self.use_amp:
                with autocast():
                    # 预测噪声
                    noise_pred = self.model(raw_norm)
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(
                        noise_pred=noise_pred,
                        noise_target=noise_target,
                        raw_input=raw_norm,
                        clean_target=clean_norm
                    )
                
                # 反向传播 (缩放梯度)
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 预测噪声
                noise_pred = self.model(raw_norm)
                
                # 计算损失
                loss, loss_dict = self.criterion(
                    noise_pred=noise_pred,
                    noise_target=noise_target,
                    raw_input=raw_norm,
                    clean_target=clean_norm
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )
                
                # 更新参数
                self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step(self.current_epoch + batch_idx / num_batches)
            
            # 更新统计
            batch_size = raw_norm.size(0)
            loss_meter.update(loss.item(), batch_size)
            noise_loss_meter.update(loss_dict['noise'].item(), batch_size)
            reconstruct_loss_meter.update(loss_dict['reconstruct'].item(), batch_size)
            
            # 日志 (每 N 个 batch)
            log_interval = self.config.get('log_interval', 50)
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch [{self.current_epoch+1}] "
                    f"Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss_meter.avg:.6f} "
                    f"(Noise: {noise_loss_meter.avg:.6f}, "
                    f"Recon: {reconstruct_loss_meter.avg:.6f}) "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                    f"Time: {elapsed:.1f}s"
                )
        
        return {
            'train_loss': loss_meter.avg,
            'train_noise_loss': noise_loss_meter.avg,
            'train_reconstruct_loss': reconstruct_loss_meter.avg
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        执行验证
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        # 损失统计
        loss_meter = AverageMeter()
        noise_loss_meter = AverageMeter()
        reconstruct_loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        mse_meter = AverageMeter()
        
        for batch in self.val_loader:
            # 移动数据到设备
            raw_norm = batch['input'].to(self.device)
            noise_target = batch['target'].to(self.device)
            clean_norm = batch['clean_norm'].to(self.device)
            mean = batch['mean'].to(self.device)
            std = batch['std'].to(self.device)
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    noise_pred = self.model(raw_norm)
                    loss, loss_dict = self.criterion(
                        noise_pred=noise_pred,
                        noise_target=noise_target,
                        raw_input=raw_norm,
                        clean_target=clean_norm
                    )
            else:
                noise_pred = self.model(raw_norm)
                loss, loss_dict = self.criterion(
                    noise_pred=noise_pred,
                    noise_target=noise_target,
                    raw_input=raw_norm,
                    clean_target=clean_norm
                )
            
            # 执行完整的去噪还原
            # Denoised = Raw_Norm - Noise_Pred (归一化域)
            denoised_norm = raw_norm - noise_pred
            
            # 还原到原始尺度 (近似)
            # 注意: 这里只是演示，实际还原需要更复杂的逆变换
            # 由于我们在归一化域比较，这里直接用归一化后的数据计算指标
            
            # 计算 PSNR (在归一化域)
            psnr = self.psnr_metric(denoised_norm, clean_norm)
            
            # 计算 MSE
            mse = torch.mean((denoised_norm - clean_norm) ** 2)
            
            # 更新统计
            batch_size = raw_norm.size(0)
            loss_meter.update(loss.item(), batch_size)
            noise_loss_meter.update(loss_dict['noise'].item(), batch_size)
            reconstruct_loss_meter.update(loss_dict['reconstruct'].item(), batch_size)
            psnr_meter.update(psnr.item(), batch_size)
            mse_meter.update(mse.item(), batch_size)
        
        return {
            'val_loss': loss_meter.avg,
            'val_noise_loss': noise_loss_meter.avg,
            'val_reconstruct_loss': reconstruct_loss_meter.avg,
            'val_psnr': psnr_meter.avg,
            'val_mse': mse_meter.avg
        }
    
    def save_checkpoint(
        self,
        is_best: bool = False,
        filename: str = 'checkpoint.pth'
    ) -> None:
        """
        保存检查点
        
        Args:
            is_best: 是否为最佳模型
            filename: 文件名
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            self.logger.info(f"保存最佳模型 (Epoch {self.current_epoch + 1})")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_psnr = checkpoint.get('best_psnr', float('-inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"加载检查点: {checkpoint_path} (Epoch {self.current_epoch})")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> None:
        """
        执行完整训练流程
        
        Args:
            num_epochs: 总训练轮数
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练
        if resume_from is not None and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        self.logger.info("=" * 60)
        self.logger.info("开始训练 SpectrogramNAFNet")
        self.logger.info("=" * 60)
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"模型参数量: {count_parameters(self.model):,}")
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        self.logger.info(f"批次大小: {self.config.get('batch_size', 32)}")
        self.logger.info(f"总轮数: {num_epochs}")
        self.logger.info("=" * 60)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            # 日志
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] 完成 "
                f"(耗时: {epoch_time:.1f}s)"
            )
            self.logger.info(
                f"  训练 - Loss: {train_metrics['train_loss']:.6f} "
                f"(Noise: {train_metrics['train_noise_loss']:.6f}, "
                f"Recon: {train_metrics['train_reconstruct_loss']:.6f})"
            )
            self.logger.info(
                f"  验证 - Loss: {val_metrics['val_loss']:.6f} "
                f"(Noise: {val_metrics['val_noise_loss']:.6f}, "
                f"Recon: {val_metrics['val_reconstruct_loss']:.6f})"
            )
            self.logger.info(
                f"  验证 - PSNR: {val_metrics['val_psnr']:.2f} dB, "
                f"MSE: {val_metrics['val_mse']:.6f}"
            )
            
            # 检查是否为最佳模型
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_psnr = val_metrics['val_psnr']
            
            # 保存检查点
            self.save_checkpoint(is_best=is_best)
            
            # 定期保存
            save_interval = self.config.get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    is_best=False,
                    filename=f'checkpoint_epoch_{epoch+1}.pth'
                )
        
        self.logger.info("=" * 60)
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        self.logger.info(f"最佳 PSNR: {self.best_psnr:.2f} dB")
        self.logger.info("=" * 60)


def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(
        description='SpectrogramNAFNet 训练脚本'
    )
    
    # 数据参数
    parser.add_argument(
        '--raw_dir', type=str, required=True,
        help='原始数据目录 (包含 train/val/test 子目录)'
    )
    parser.add_argument(
        '--clean_dir', type=str, required=True,
        help='干净数据目录 (包含 train/val/test 子目录)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help='输出目录'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='批次大小'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='学习率'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-2,
        help='权重衰减'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='数据加载工作进程数'
    )
    
    # 模型参数
    parser.add_argument(
        '--base_channels', type=int, default=32,
        help='基础通道数'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout 概率'
    )
    
    # 损失函数参数
    parser.add_argument(
        '--noise_weight', type=float, default=1.0,
        help='噪声损失权重'
    )
    parser.add_argument(
        '--reconstruct_weight', type=float, default=1.0,
        help='重建损失权重'
    )
    
    # 调度器参数
    parser.add_argument(
        '--T_0', type=int, default=10,
        help='CosineAnnealingWarmRestarts 的 T_0'
    )
    parser.add_argument(
        '--T_mult', type=int, default=2,
        help='CosineAnnealingWarmRestarts 的 T_mult'
    )
    
    # 其他参数
    parser.add_argument(
        '--resume', type=str, default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--no_amp', action='store_true',
        help='禁用混合精度训练'
    )
    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='梯度裁剪阈值'
    )
    parser.add_argument(
        '--log_interval', type=int, default=50,
        help='日志打印间隔 (batch)'
    )
    parser.add_argument(
        '--save_interval', type=int, default=10,
        help='检查点保存间隔 (epoch)'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置
    config = {
        'raw_dir': args.raw_dir,
        'clean_dir': args.clean_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'base_channels': args.base_channels,
        'dropout': args.dropout,
        'noise_weight': args.noise_weight,
        'reconstruct_weight': args.reconstruct_weight,
        'T_0': args.T_0,
        'T_mult': args.T_mult,
        'use_amp': not args.no_amp,
        'grad_clip': args.grad_clip,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'seed': args.seed
    }
    
    # 数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        raw_base_dir=args.raw_dir,
        clean_base_dir=args.clean_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 模型
    model = SpectrogramNAFNet(
        in_channels=2,
        out_channels=2,
        base_channels=args.base_channels,
        num_blocks=[2, 2, 4, 8],
        bottleneck_blocks=4,
        dropout_rate=args.dropout
    ).to(device)
    
    # 损失函数
    criterion = CompositeLoss(
        noise_weight=args.noise_weight,
        reconstruct_weight=args.reconstruct_weight
    )
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # 开始训练
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == '__main__':
    main()
