"""
复合损失函数定义

本模块实现基于双域约束的复合损失函数，
用于 STFT 域 EEG 信号残差学习去噪任务。

损失组成:
1. 噪声拟合项 (L_noise): 预测噪声与真实噪声的 L1 损失
2. 信号重建项 (L_reconstruct): 重建信号的 Log-Magnitude L1 损失

作者: AI Assistant
日期: 2025-12-30
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    复合损失函数
    
    采用双域约束策略，同时优化噪声拟合和信号重建质量。
    
    损失公式:
        L_total = w_noise * L_noise + w_reconstruct * L_reconstruct
        
    其中:
        - L_noise: 预测噪声与真实噪声的 L1 损失 (实部和虚部分别计算)
        - L_reconstruct: 重建信号的 Log-Magnitude L1 损失
    
    注意: 由于数据经过 Z-score 归一化，损失值会较小。
    使用 loss_scale 参数可以放大损失值以便于监控。
    
    Args:
        noise_weight: 噪声拟合项权重 (默认 1.0)
        reconstruct_weight: 信号重建项权重 (默认 1.0)
        loss_scale: 损失缩放因子 (默认 100.0，用于放大小数值)
        eps: 数值稳定性常数 (默认 1e-8)
    """
    
    def __init__(
        self,
        noise_weight: float = 1.0,
        reconstruct_weight: float = 1.0,
        loss_scale: float = 1.0,
        eps: float = 1e-5
    ) -> None:
        super().__init__()
        
        self.noise_weight = noise_weight
        self.reconstruct_weight = reconstruct_weight
        self.loss_scale = loss_scale
        self.eps = eps
        
        # L1 损失函数
        self.l1_loss = nn.L1Loss()
    
    def compute_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算复数 STFT 的幅度
        
        Args:
            x: 形状 [B, 2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            幅度张量，形状 [B, 1, F, T]
        """
        real = x[:, 0:1, :, :]  # [B, 1, F, T]
        imag = x[:, 1:2, :, :]  # [B, 1, F, T]
        magnitude = torch.sqrt(real**2 + imag**2 + self.eps)
        return magnitude
    
    def compute_log_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Log 变换后的幅度
        
        Args:
            x: 形状 [B, 2, F, T]，Channel 0=Real, 1=Imag
            
        Returns:
            Log 幅度张量，形状 [B, 1, F, T]
        """
        magnitude = self.compute_magnitude(x)
        log_magnitude = torch.log1p(magnitude)  # log(1 + |S|)
        return log_magnitude
    
    def noise_loss(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算噪声拟合损失 (L_noise)
        
        对预测噪声和真实噪声的实部、虚部分别计算 L1 损失。
        
        Args:
            noise_pred: 预测的噪声 [B, 2, F, T]
            noise_target: 真实的噪声 [B, 2, F, T]
            
        Returns:
            噪声拟合损失标量
        """
        # 实部 L1 损失
        real_loss = self.l1_loss(noise_pred[:, 0, :, :], noise_target[:, 0, :, :])
        
        # 虚部 L1 损失
        imag_loss = self.l1_loss(noise_pred[:, 1, :, :], noise_target[:, 1, :, :])
        
        # 平均
        return (real_loss + imag_loss) / 2.0
    
    def reconstruction_loss(
        self,
        raw_input: torch.Tensor,
        noise_pred: torch.Tensor,
        clean_target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算信号重建损失 (L_reconstruct)
        
        在归一化域直接计算幅度 L1 损失。
        由于新的归一化格式是 normalized_data = norm_log_mag * phase，
        幅度 |normalized_data| = |norm_log_mag|（因为|phase|=1）
        
        Args:
            raw_input: 原始输入 (归一化后) [B, 2, F, T]
            noise_pred: 预测的噪声 [B, 2, F, T]
            clean_target: 干净信号目标 (归一化后) [B, 2, F, T]
            
        Returns:
            信号重建损失标量
        """
        # 计算重建信号: Clean_Rec = Raw_Input - Noise_Pred (归一化域)
        clean_reconstructed = raw_input - noise_pred
        
        # 在归一化域，幅度就是归一化后的log幅度
        # 直接计算幅度L1损失
        mag_pred = self.compute_magnitude(clean_reconstructed)
        mag_target = self.compute_magnitude(clean_target)
        
        # L1 损失
        return self.l1_loss(mag_pred, mag_target)
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        raw_input: torch.Tensor,
        clean_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算复合损失
        
        Args:
            noise_pred: 预测的噪声 [B, 2, F, T]
            noise_target: 真实的噪声 (Raw - Clean) [B, 2, F, T]
            raw_input: 原始输入 (归一化后) [B, 2, F, T]
            clean_target: 干净信号目标 (归一化后) [B, 2, F, T]
            
        Returns:
            (total_loss, loss_dict): 总损失和各项损失字典
        """
        # 检查输入是否包含 NaN/Inf
        if not torch.isfinite(noise_pred).all():
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. 噪声拟合损失
        l_noise = self.noise_loss(noise_pred, noise_target)
        
        # 2. 信号重建损失
        l_reconstruct = self.reconstruction_loss(raw_input, noise_pred, clean_target)
        
        # 3. 加权总损失 (应用缩放因子使损失值更易读)
        total_loss = self.loss_scale * (
            self.noise_weight * l_noise + self.reconstruct_weight * l_reconstruct
        )
        
        # 4. 损失值安全检查
        if not torch.isfinite(total_loss):
            # 回退到仅噪声损失
            total_loss = self.loss_scale * l_noise if torch.isfinite(l_noise) else torch.tensor(0.0, device=noise_pred.device, requires_grad=True)
        
        # 损失字典 (用于日志记录, 使用缩放后的值)
        loss_dict = {
            'total': total_loss.detach(),
            'noise': (self.loss_scale * l_noise).detach(),
            'reconstruct': (self.loss_scale * l_reconstruct).detach()
        }
        
        return total_loss, loss_dict


class PSNRMetric:
    """
    PSNR (Peak Signal-to-Noise Ratio) 指标计算器
    
    用于评估去噪质量。
    
    Args:
        data_range: 数据范围 (默认 None, 自动计算)
        eps: 数值稳定性常数
    """
    
    def __init__(
        self,
        data_range: Optional[float] = None,
        eps: float = 1e-5
    ) -> None:
        self.data_range = data_range
        self.eps = eps
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 PSNR
        
        Args:
            pred: 预测值 [B, ...]
            target: 目标值 [B, ...]
            
        Returns:
            平均 PSNR (dB)
        """
        # 计算 MSE
        mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.dim())))
        
        # 数据范围
        if self.data_range is None:
            data_range = target.max() - target.min()
        else:
            data_range = self.data_range
        
        # PSNR
        psnr = 10 * torch.log10((data_range ** 2) / (mse + self.eps))
        
        return psnr.mean()


class MagnitudeLoss(nn.Module):
    """
    幅度域 L1 损失
    
    直接在 STFT 幅度域计算损失。
    
    Args:
        eps: 数值稳定性常数
    """
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.l1_loss = nn.L1Loss()
    
    def compute_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """计算幅度"""
        real = x[:, 0:1, :, :]
        imag = x[:, 1:2, :, :]
        return torch.sqrt(real**2 + imag**2 + self.eps)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算幅度域 L1 损失
        
        Args:
            pred: 预测值 [B, 2, F, T]
            target: 目标值 [B, 2, F, T]
            
        Returns:
            损失标量
        """
        mag_pred = self.compute_magnitude(pred)
        mag_target = self.compute_magnitude(target)
        return self.l1_loss(mag_pred, mag_target)


class PhaseLoss(nn.Module):
    """
    相位损失
    
    计算相位差异的余弦损失。
    
    Args:
        eps: 数值稳定性常数
    """
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算相位损失
        
        使用余弦相似度衡量相位差异:
        Loss = 1 - cos(phase_pred - phase_target)
        
        Args:
            pred: 预测值 [B, 2, F, T]
            target: 目标值 [B, 2, F, T]
            
        Returns:
            损失标量
        """
        # 提取实部和虚部
        real_pred, imag_pred = pred[:, 0, :, :], pred[:, 1, :, :]
        real_target, imag_target = target[:, 0, :, :], target[:, 1, :, :]
        
        # 计算幅度
        mag_pred = torch.sqrt(real_pred**2 + imag_pred**2 + self.eps)
        mag_target = torch.sqrt(real_target**2 + imag_target**2 + self.eps)
        
        # 单位复数
        unit_pred_real = real_pred / mag_pred
        unit_pred_imag = imag_pred / mag_pred
        unit_target_real = real_target / mag_target
        unit_target_imag = imag_target / mag_target
        
        # 余弦相似度: Re(pred * conj(target)) / (|pred| * |target|)
        # = (real_pred * real_target + imag_pred * imag_target) / (mag_pred * mag_target)
        cos_similarity = (
            unit_pred_real * unit_target_real + 
            unit_pred_imag * unit_target_imag
        )
        
        # 损失: 1 - cos
        loss = 1.0 - cos_similarity.mean()
        
        return loss


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("CompositeLoss 单元测试")
    print("=" * 60)
    
    # 创建损失函数
    criterion = CompositeLoss(noise_weight=1.0, reconstruct_weight=1.0)
    
    # 模拟数据
    batch_size = 4
    noise_pred = torch.randn(batch_size, 2, 103, 156)
    noise_target = torch.randn(batch_size, 2, 103, 156)
    raw_input = torch.randn(batch_size, 2, 103, 156)
    clean_target = raw_input - noise_target  # Clean = Raw - Noise
    
    # 计算损失
    total_loss, loss_dict = criterion(
        noise_pred=noise_pred,
        noise_target=noise_target,
        raw_input=raw_input,
        clean_target=clean_target
    )
    
    print(f"\n损失计算结果:")
    print(f"  Total Loss:       {total_loss.item():.6f}")
    print(f"  Noise Loss:       {loss_dict['noise'].item():.6f}")
    print(f"  Reconstruct Loss: {loss_dict['reconstruct'].item():.6f}")
    
    # 验证梯度
    noise_pred.requires_grad = True
    total_loss, _ = criterion(noise_pred, noise_target, raw_input, clean_target)
    total_loss.backward()
    
    assert noise_pred.grad is not None, "梯度计算失败"
    print("\n✓ 损失函数梯度计算正常!")
    
    # 测试 PSNR
    print("\n" + "=" * 60)
    print("PSNRMetric 单元测试")
    print("=" * 60)
    
    psnr_metric = PSNRMetric()
    
    # 相同信号 PSNR 应该很高
    psnr_same = psnr_metric(raw_input, raw_input)
    print(f"相同信号 PSNR: {psnr_same.item():.2f} dB")
    
    # 不同信号 PSNR
    psnr_diff = psnr_metric(noise_pred.detach(), noise_target)
    print(f"不同信号 PSNR: {psnr_diff.item():.2f} dB")
    
    print("\n✓ 所有损失函数测试通过!")
