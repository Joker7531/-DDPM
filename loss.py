"""
多分辨率STFT损失函数
Multi-Resolution STFT Loss for time-series generation
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率短时傅里叶变换损失
    计算预测信号与真实信号在不同FFT分辨率下的幅度谱L1距离
    
    Args:
        fft_sizes: FFT窗口大小列表
        hop_sizes: 跳跃步长列表
        win_lengths: 窗口长度列表（默认与fft_sizes相同）
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [64, 128, 256, 512, 1024, 2048],
        hop_sizes: List[int] = [16, 32, 64, 128, 256, 512],
        win_lengths: List[int] = None
    ):
        super().__init__()
        
        if win_lengths is None:
            win_lengths = fft_sizes
            
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "FFT sizes, hop sizes, and window lengths must have the same length"
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        # 预注册Hann窗口（避免每次前向传播时重新创建）
        for fft_size, win_length in zip(fft_sizes, win_lengths):
            self.register_buffer(
                f'window_{fft_size}',
                torch.hann_window(win_length)
            )
    
    def stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int
    ) -> torch.Tensor:
        """
        计算短时傅里叶变换
        
        Args:
            x: 输入信号 [B, 1, T]
            fft_size: FFT大小
            hop_size: 跳跃步长
            win_length: 窗口长度
            
        Returns:
            幅度谱 [B, freq_bins, time_frames]
        """
        # 获取预注册的窗口
        window = getattr(self, f'window_{fft_size}')
        
        # 确保输入是2D: [B, T]
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # 计算STFT
        stft_result = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            normalized=False,
            center=True
        )
        
        # 计算幅度谱
        magnitude = torch.abs(stft_result)  # [B, freq_bins, time_frames]
        
        return magnitude
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算多分辨率STFT损失
        
        Args:
            y_pred: 预测信号 [B, 1, 2048]
            y_true: 真实信号 [B, 1, 2048]
            
        Returns:
            total_loss: 总损失（所有分辨率的平均）
            loss_dict: 每个分辨率的损失字典（用于日志记录）
        """
        total_loss = 0.0
        loss_dict = {}
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # 计算预测和真实信号的STFT幅度谱
            mag_pred = self.stft(y_pred, fft_size, hop_size, win_length)
            mag_true = self.stft(y_true, fft_size, hop_size, win_length)
            
            # 计算L1距离
            spectral_loss = torch.mean(torch.abs(mag_pred - mag_true))
            
            # 累加到总损失
            total_loss += spectral_loss
            
            # 记录每个分辨率的损失
            loss_dict[f'stft_{fft_size}'] = spectral_loss.item()
        
        # 返回平均损失
        avg_loss = total_loss / len(self.fft_sizes)
        
        return avg_loss, loss_dict


if __name__ == "__main__":
    # 测试代码
    batch_size = 4
    length = 2048
    
    # 创建损失函数
    stft_loss = MultiResolutionSTFTLoss()
    
    # 生成随机测试数据
    y_pred = torch.randn(batch_size, 1, length)
    y_true = torch.randn(batch_size, 1, length)
    
    # 计算损失
    loss, loss_dict = stft_loss(y_pred, y_true)
    
    print(f"Total STFT Loss: {loss.item():.6f}")
    print("\nPer-resolution losses:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
