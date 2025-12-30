"""
SpectrogramNAFNet 模型架构定义

本模块实现基于 NAFNet (Nonlinear Activation Free Network) 的 U-Net 变体，
专为 STFT 域 EEG 信号去噪设计。

核心特点:
- 无非线性激活函数 (No ReLU/GELU/Sigmoid)
- SimpleGate 门控机制
- 简化通道注意力 (SCA)
- 奇数维度自适应填充

作者: AI Assistant
日期: 2025-12-30
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization
    
    针对图像/频谱图的 LayerNorm 实现，
    沿通道维度进行归一化。
    
    Args:
        num_channels: 输入通道数
        eps: 数值稳定性常数
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            归一化后的张量 [B, C, H, W]
        """
        # 计算均值和方差 (沿 C, H, W 维度)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # 应用可学习的缩放和偏移
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    """
    Simple Gate 门控模块
    
    将通道分为两半，执行元素级乘法实现门控。
    这是 NAFNet 的核心组件，替代传统激活函数。
    
    公式: output = x1 * x2
    其中 x1, x2 是输入沿通道维度分割的两半
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]，C 必须为偶数
            
        Returns:
            门控后的张量 [B, C//2, H, W]
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    简化通道注意力 (SCA - Simplified Channel Attention)
    
    使用全局平均池化和 1x1 卷积实现轻量级通道注意力。
    
    Args:
        num_channels: 输入通道数
    """
    
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            注意力加权后的张量 [B, C, H, W]
        """
        # 全局平均池化
        y = self.pool(x)  # [B, C, 1, 1]
        # 1x1 卷积生成通道权重
        y = self.conv(y)  # [B, C, 1, 1]
        # 通道加权
        return x * y


class NAFBlock(nn.Module):
    """
    NAFNet 基本构建块
    
    无非线性激活函数的残差块，使用 SimpleGate 和 SCA 实现特征变换。
    
    结构顺序:
    1. LayerNorm2d
    2. Conv2d (1x1, 升维 2倍)
    3. Conv2d (3x3, Depthwise)
    4. SimpleGate (通道减半)
    5. SCA (通道注意力)
    6. Conv2d (1x1, 降维)
    7. Dropout
    8. Residual Add
    
    Args:
        in_channels: 输入通道数
        dropout_rate: Dropout 概率
        expansion_ratio: 通道扩展比率
    """
    
    def __init__(
        self,
        in_channels: int,
        dropout_rate: float = 0.0,
        expansion_ratio: int = 2
    ) -> None:
        super().__init__()
        
        # 扩展后的通道数 (用于 SimpleGate 前)
        expanded_channels = in_channels * expansion_ratio * 2  # *2 因为 SimpleGate 会减半
        
        # LayerNorm
        self.norm = LayerNorm2d(in_channels)
        
        # 1x1 卷积升维
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1)
        
        # 3x3 Depthwise 卷积
        self.dwconv = nn.Conv2d(
            expanded_channels, expanded_channels,
            kernel_size=3, padding=1, groups=expanded_channels
        )
        
        # SimpleGate
        self.gate = SimpleGate()
        
        # SCA (SimpleGate 后通道数减半)
        self.sca = SimplifiedChannelAttention(expanded_channels // 2)
        
        # 1x1 卷积降维
        self.conv2 = nn.Conv2d(expanded_channels // 2, in_channels, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Beta 参数 (用于残差缩放)
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C, H, W]
        """
        identity = x
        
        # LayerNorm
        x = self.norm(x)
        
        # 1x1 升维
        x = self.conv1(x)
        
        # 3x3 Depthwise
        x = self.dwconv(x)
        
        # SimpleGate
        x = self.gate(x)
        
        # SCA
        x = self.sca(x)
        
        # 1x1 降维
        x = self.conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 残差连接
        return identity + x * self.beta


class Downsample(nn.Module):
    """
    下采样模块
    
    使用 PixelUnshuffle 实现 2x 下采样。
    
    Args:
        in_channels: 输入通道数
    """
    
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # PixelUnshuffle 将空间尺寸减半，通道数增加 4 倍
        self.unshuffle = nn.PixelUnshuffle(2)
        # 1x1 卷积调整通道数
        self.conv = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            下采样后的张量 [B, 2C, H//2, W//2]
        """
        x = self.unshuffle(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    上采样模块
    
    使用 PixelShuffle 实现 2x 上采样。
    
    Args:
        in_channels: 输入通道数
    """
    
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # 1x1 卷积准备通道数
        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        # PixelShuffle 将通道数减少 4 倍，空间尺寸增加 2 倍
        self.shuffle = nn.PixelShuffle(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            上采样后的张量 [B, C//2, H*2, W*2]
        """
        x = self.conv(x)
        x = self.shuffle(x)
        return x


class SpectrogramNAFNet(nn.Module):
    """
    SpectrogramNAFNet: 基于 NAFNet 的 STFT 域去噪网络
    
    采用 U-Net 结构的编码器-解码器架构，使用 NAFBlock 作为基本构建单元。
    专为处理 STFT 频谱图设计，包含奇数维度适配器。
    
    Args:
        in_channels: 输入通道数 (默认 2: Real + Imag)
        out_channels: 输出通道数 (默认 2: Real + Imag)
        base_channels: 基础通道数 (默认 32)
        num_blocks: 每层的 NAFBlock 数量 (默认 [2, 2, 4, 8])
        bottleneck_blocks: 瓶颈层 NAFBlock 数量 (默认 4)
        dropout_rate: Dropout 概率 (默认 0.0)
    """
    
    # 用于确保维度可被整除的目标尺寸
    DIVISOR: int = 16
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        num_blocks: List[int] = None,
        bottleneck_blocks: int = 4,
        dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 4, 8]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_levels = len(num_blocks)
        
        # 输入投影
        self.intro = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        channels = base_channels
        for i, num_block in enumerate(num_blocks):
            # NAFBlock 堆叠
            encoder_blocks = nn.Sequential(
                *[NAFBlock(channels, dropout_rate) for _ in range(num_block)]
            )
            self.encoders.append(encoder_blocks)
            
            # 下采样 (最后一层不下采样)
            if i < len(num_blocks) - 1:
                self.downs.append(Downsample(channels))
                channels *= 2
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            *[NAFBlock(channels, dropout_rate) for _ in range(bottleneck_blocks)]
        )
        
        # 解码器
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(len(num_blocks) - 1, -1, -1):
            if i > 0:
                # 上采样
                self.ups.append(Upsample(channels))
                channels //= 2
                
                # 融合卷积 (处理 skip connection)
                self.fusions.append(
                    nn.Conv2d(channels * 2, channels, kernel_size=1)
                )
            
            # NAFBlock 堆叠
            decoder_blocks = nn.Sequential(
                *[NAFBlock(channels, dropout_rate) for _ in range(num_blocks[i])]
            )
            self.decoders.append(decoder_blocks)
        
        # 输出投影
        self.outro = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def _pad_to_divisible(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        填充输入使其维度可被 DIVISOR 整除
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            (padded_x, pad_info): 填充后的张量和填充信息
        """
        _, _, h, w = x.shape
        
        # 计算需要填充的量
        pad_h = (self.DIVISOR - h % self.DIVISOR) % self.DIVISOR
        pad_w = (self.DIVISOR - w % self.DIVISOR) % self.DIVISOR
        
        # 使用 reflect 模式填充
        if pad_h > 0 or pad_w > 0:
            # F.pad 的顺序: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        return x, (0, pad_h, 0, pad_w)
    
    def _crop_to_original(
        self,
        x: torch.Tensor,
        pad_info: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        裁剪输出回原始尺寸
        
        Args:
            x: 填充后的张量 [B, C, H', W']
            pad_info: 填充信息 (pad_left, pad_h, pad_top, pad_w)
            
        Returns:
            裁剪后的张量 [B, C, H, W]
        """
        _, pad_h, _, pad_w = pad_info
        
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        实现奇数维度适配：
        1. Pad: 填充输入至可被16整除
        2. Forward: 通过 U-Net 网络
        3. Crop: 裁剪输出回原始尺寸
        
        Args:
            x: 输入张量 [B, 2, F, T]，F=103, T=156
            
        Returns:
            输出张量 [B, 2, F, T]，预测的噪声残差
        """
        # 保存原始尺寸
        original_shape = x.shape
        
        # 1. 填充到可被16整除
        x, pad_info = self._pad_to_divisible(x)
        
        # 2. 输入投影
        x = self.intro(x)
        
        # 3. 编码器 (保存 skip connections)
        skip_connections = []
        for i, (encoder, down) in enumerate(zip(self.encoders[:-1], self.downs)):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x)
        
        # 最后一层编码器 (不下采样)
        x = self.encoders[-1](x)
        
        # 4. 瓶颈层
        x = self.bottleneck(x)
        
        # 5. 解码器
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                # 上采样
                x = self.ups[i - 1](x)
                # Skip connection
                skip = skip_connections[-(i)]
                x = torch.cat([x, skip], dim=1)
                # 融合
                x = self.fusions[i - 1](x)
            x = decoder(x)
        
        # 6. 输出投影
        x = self.outro(x)
        
        # 7. 裁剪回原始尺寸
        x = self._crop_to_original(x, pad_info)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("SpectrogramNAFNet 单元测试")
    print("=" * 60)
    
    # 创建模型
    model = SpectrogramNAFNet(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        num_blocks=[2, 2, 4, 8],
        bottleneck_blocks=4
    )
    
    print(f"\n模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    # 输入: [B, 2, 103, 156] (频率维度 103 是奇数)
    batch_size = 4
    x = torch.randn(batch_size, 2, 103, 156)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状与输入相同
    assert y.shape == x.shape, f"形状不匹配: 输入 {x.shape}, 输出 {y.shape}"
    
    print("\n✓ 模型测试通过!")
    
    # 测试 NAFBlock
    print("\n" + "=" * 60)
    print("NAFBlock 单元测试")
    print("=" * 60)
    
    block = NAFBlock(32)
    x_block = torch.randn(2, 32, 16, 16)
    y_block = block(x_block)
    
    print(f"NAFBlock 输入: {x_block.shape}")
    print(f"NAFBlock 输出: {y_block.shape}")
    
    assert x_block.shape == y_block.shape, "NAFBlock 输出形状应与输入相同"
    print("✓ NAFBlock 测试通过!")
