"""
条件扩散波网络 (Conditional DiffWave)
基于膨胀卷积的1D ResNet架构，用于EEG去噪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码
    用于将时间步t编码为高维特征向量
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 [B] or [B, 1]
            
        Returns:
            位置编码 [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        
        # 计算频率
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 确保t是1维的
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # 广播乘法
        emb = t[:, None] * emb[None, :]
        
        # 拼接sin和cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    """
    残差块，使用膨胀卷积和门控激活
    
    Args:
        residual_channels: 残差通道数
        dilation: 膨胀系数
        time_emb_dim: 时间步嵌入维度
    """
    
    def __init__(
        self,
        residual_channels: int = 256,
        dilation: int = 1,
        time_emb_dim: int = 512
    ):
        super().__init__()
        
        # 膨胀卷积层
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,  # 为门控激活预留双倍通道
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        
        # 时间步投影层
        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # Swish激活
            nn.Linear(time_emb_dim, 2 * residual_channels)
        )
        
        # 输出投影（分为残差和跳跃连接）
        self.output_projection = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=1
        )
        
        # Instance Normalization (每个样本独立归一化)
        self.instance_norm = nn.InstanceNorm1d(residual_channels, affine=True)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [B, C, T]
            time_emb: 时间步嵌入 [B, time_emb_dim]
            
        Returns:
            residual_output: 残差输出 [B, C, T]
            skip_output: 跳跃连接输出 [B, C, T]
        """
        # Instance Normalization
        h = self.instance_norm(x)
        
        # 膨胀卷积
        h = self.dilated_conv(h)
        
        # 时间步调制
        time_emb_proj = self.time_mlp(time_emb)[:, :, None]  # [B, 2*C, 1]
        h = h + time_emb_proj
        
        # 门控激活: tanh(h1) * sigmoid(h2)
        gate, filter_val = h.chunk(2, dim=1)
        h = torch.tanh(gate) * torch.sigmoid(filter_val)
        
        # 输出投影
        h = self.output_projection(h)
        
        # 分离残差和跳跃连接
        residual, skip = h.chunk(2, dim=1)
        
        # 残差连接
        residual_output = (x + residual) / math.sqrt(2.0)  # 缩放以稳定训练
        
        return residual_output, skip


class ConditionalDiffWave(nn.Module):
    """
    条件扩散波网络
    
    Args:
        in_channels: 输入通道数 (2: noisy + condition)
        out_channels: 输出通道数 (1: predicted noise)
        residual_channels: 残差块通道数
        num_layers: 残差块层数
        dilation_cycle: 膨胀系数循环长度
        time_emb_dim: 时间步嵌入维度
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        residual_channels: int = 256,
        num_layers: int = 30,
        dilation_cycle: int = 10,  # [1, 2, 4, ..., 512] 共10个，循环3次
        time_emb_dim: int = 512,
        condition_dropout: float = 0.1  # 条件Dropout概率
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.num_layers = num_layers
        self.condition_dropout = condition_dropout
        
        # 时间步编码
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim // 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 输入投影
        self.input_projection = nn.Conv1d(
            in_channels,
            residual_channels,
            kernel_size=1
        )
        
        # 条件Dropout（在训练时随机丢弃条件信息）
        self.condition_dropout_layer = nn.Dropout(condition_dropout)
        
        # 构建残差块
        self.residual_layers = nn.ModuleList()
        for i in range(num_layers):
            # 膨胀系数: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] 循环
            dilation = 2 ** (i % dilation_cycle)
            self.residual_layers.append(
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=dilation,
                    time_emb_dim=time_emb_dim
                )
            )
        
        # 跳跃连接输出头
        self.skip_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, out_channels, kernel_size=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 带噪信号 x_t [B, 1, 2048]
            t: 扩散时间步 [B] or [B, 1]
            condition: 条件信号 (raw EEG) [B, 1, 2048]
            
        Returns:
            predicted_noise: 预测的噪声 epsilon [B, 1, 2048]
        """
        # 拼接噪声信号和条件信号
        if condition is not None:
            # 在训练时应用条件Dropout
            if self.training and self.condition_dropout > 0:
                condition = self.condition_dropout_layer(condition)
            x = torch.cat([x, condition], dim=1)  # [B, 2, 2048]
        
        # 时间步嵌入
        time_emb = self.time_pos_emb(t)  # [B, time_emb_dim//2]
        time_emb = self.time_mlp(time_emb)  # [B, time_emb_dim]
        
        # 输入投影
        h = self.input_projection(x)  # [B, residual_channels, 2048]
        
        # 收集跳跃连接
        skip_connections = []
        
        # 通过所有残差块
        for layer in self.residual_layers:
            h, skip = layer(h, time_emb)
            skip_connections.append(skip)
        
        # 求和所有跳跃连接
        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)
        
        # 输出头
        output = self.skip_projection(skip_sum)  # [B, 1, 2048]
        
        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ConditionalDiffWave(
        in_channels=2,
        out_channels=1,
        residual_channels=256,
        num_layers=30,
        dilation_cycle=10,
        time_emb_dim=512
    ).to(device)
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # 测试前向传播
    batch_size = 2
    x_noisy = torch.randn(batch_size, 1, 2048).to(device)
    condition = torch.randn(batch_size, 1, 2048).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    with torch.no_grad():
        output = model(x_noisy, t, condition)
    
    print(f"\nInput shape: {x_noisy.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Time step shape: {t.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算感受野
    max_dilation = 2 ** 9  # 512
    kernel_size = 3
    receptive_field = 1 + (model.num_layers // 10) * sum(
        2 * (2 ** i) * (kernel_size - 1) for i in range(10)
    )
    print(f"\nTheoretical receptive field: {receptive_field} samples")
