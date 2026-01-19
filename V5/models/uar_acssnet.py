"""
UAR-ACSSNet: Unified Artifact Removal with Axis-Conditioned Selective Scan Network
单通道 EEG 去伪影端到端模型

v3.0: 使用 MDTA (Multi-Dconv Head Transposed Attention) 变体
      - 双轴并行注意力：时间全局性 + 频率全局性
      - 基于 Restormer (CVPR 2022) 的转置注意力机制
      - 深度可分离卷积增强局部特征
      - 保留自适应门控融合策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import sys
from pathlib import Path
from einops import rearrange, repeat

# 确保可以导入 signal_processing
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from signal_processing import STFTProcessor


# ================================
# 0) MDTA 核心实现 (Multi-Dconv Head Transposed Attention)
# ================================

class DepthwiseConv2d(nn.Module):
    """深度可分离卷积，用于局部特征增强"""
    def __init__(self, channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=bias,
        )
    
    def forward(self, x):
        return self.dw_conv(x)


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA)
    
    基于 Restormer (CVPR 2022) 的设计，核心创新点：
    1. 转置注意力：在通道维度计算注意力，而非空间维度
       - 计算复杂度: O(C²) vs O((H×W)²)
       - 更适合处理高分辨率特征图
    2. 深度可分离卷积：增强局部特征，弥补全局注意力对局部的不足
    3. 多头机制：不同头关注不同的通道子空间
    
    对于 EEG 时频图 (T=33, F=101):
    - 标准注意力: O(33×101)² = O(11M)
    - 转置注意力: O(C²) ≈ O(4K) (C=64)
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        bias: 是否使用偏置
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 投影 (1x1 conv)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        
        # 深度可分离卷积增强 Q, K, V 的局部特征
        self.qkv_dwconv = DepthwiseConv2d(channels * 3, kernel_size=3, bias=bias)
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 对于时频图是 (B, C, T, F)
        
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Q, K, V 投影 + 深度卷积增强
        qkv = self.qkv_dwconv(self.qkv(x))  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # 各 (B, C, H, W)
        
        # 重排为多头格式
        # (B, C, H, W) -> (B, heads, head_dim, H*W)
        q = rearrange(q, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        
        # L2 归一化 Q, K（提升稳定性）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 转置注意力：在通道维度计算
        # attn: (B, heads, head_dim, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # 应用注意力到 V
        # out: (B, heads, head_dim, H*W)
        out = attn @ v
        
        # 重排回原始形状
        out = rearrange(out, 'b h d (x y) -> b (h d) x y', x=H, y=W)
        
        # 输出投影
        out = self.proj(out)
        
        return out


class AxisMDTA(nn.Module):
    """
    轴向 MDTA - 专门针对时频图的单轴注意力
    
    设计思路:
    1. 沿指定轴（时间或频率）展开为序列
    2. 在该轴上应用转置注意力
    3. 恢复原始维度
    
    优势：
    - 计算效率更高：分解为两个独立的 1D 注意力
    - 更好的轴向特化：时间轴和频率轴有不同的特性
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        axis: str = 'freq',  # 'freq' or 'time'
        bias: bool = False,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.axis = axis
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 输入归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Q, K, V 投影
        self.to_qkv = nn.Linear(channels, channels * 3, bias=bias)
        
        # 深度可分离卷积增强（1D 版本）
        # 对于时间轴：沿 T 方向卷积
        # 对于频率轴：沿 F 方向卷积
        self.qkv_dwconv = nn.Conv1d(
            channels * 3, channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        
        # 输出投影
        self.proj = nn.Linear(channels, channels, bias=bias)
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        
        # FFN（使用深度可分离卷积的门控 FFN）
        hidden_dim = int(channels * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        self._sanity_checked = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, F)
        
        Returns:
            out: (B, C, T, F)
        """
        B, C, T, F_dim = x.shape
        identity = x
        
        # 根据轴向重排
        if self.axis == 'freq':
            # 沿频率轴注意力: 每个时间帧独立处理
            # (B, C, T, F) -> (B*T, F, C)
            x = rearrange(x, 'b c t f -> (b t) f c')
            seq_len = F_dim
            batch_size = B * T
        else:  # 'time'
            # 沿时间轴注意力: 每个频率 bin 独立处理
            # (B, C, T, F) -> (B*F, T, C)
            x = rearrange(x, 'b c t f -> (b f) t c')
            seq_len = T
            batch_size = B * F_dim
        
        # 归一化
        x_norm = self.norm1(x)
        
        # Q, K, V 投影
        qkv = self.to_qkv(x_norm)  # (batch, seq, 3*C)
        
        # 深度卷积增强（需要转置）
        qkv = rearrange(qkv, 'b s c -> b c s')
        qkv = self.qkv_dwconv(qkv)
        qkv = rearrange(qkv, 'b c s -> b s c')
        
        # 分割 Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # 各 (batch, seq, C)
        
        # 多头重排: (batch, seq, C) -> (batch, heads, seq, head_dim)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # L2 归一化（提升数值稳定性）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 转置注意力: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        # -> (batch, heads, seq, seq)
        # 注意：这里是标准注意力，但序列长度小（T=33 或 F=101）
        # 转置注意力的优势在全分辨率 2D 时更明显
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = attn @ v  # (batch, heads, seq, head_dim)
        
        # 合并多头
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        # 输出投影
        out = self.proj(out)
        out = self.dropout(out)
        
        # 残差连接
        out = out + x
        
        # FFN
        out = out + self.ffn(self.norm2(out))
        
        # 恢复原始维度
        if self.axis == 'freq':
            out = rearrange(out, '(b t) f c -> b c t f', b=B, t=T)
        else:
            out = rearrange(out, '(b f) t c -> b c t f', b=B, f=F_dim)
        
        # Sanity check（首次）
        if not self._sanity_checked and self.training:
            self._sanity_check(identity, out)
            self._sanity_checked = True
        
        return out
    
    def _sanity_check(self, x_in, x_out):
        """验证注意力模块工作正常"""
        with torch.no_grad():
            # 检查输出统计
            in_std = x_in.std().item()
            out_std = x_out.std().item()
            print(f"    [AxisMDTA/{self.axis}] Input std: {in_std:.4f}, Output std: {out_std:.4f}")
            
            # 检查温度参数
            temp_mean = self.temperature.mean().item()
            print(f"    [AxisMDTA/{self.axis}] Temperature mean: {temp_mean:.4f}")


class DualAxisMDTA(nn.Module):
    """
    双轴并行 MDTA - 同时处理时间和频率轴
    
    核心架构：
    1. 输入分解为两个并行流
    2. FreqMDTA: 捕捉频率全局性（跨频率 bin 的关系）
    3. TimeMDTA: 捕捉时间全局性（跨时间帧的关系）
    4. 自适应门控融合
    
    相比 Mamba 的优势：
    - 小序列（T=33, F=101）上注意力更高效
    - 全局感受野，一次前向即可看到整个序列
    - 可解释性更强
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        bias: bool = False,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        
        # 频率轴注意力
        self.freq_attn = AxisMDTA(
            channels=channels,
            num_heads=num_heads,
            axis='freq',
            bias=bias,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )
        
        # 时间轴注意力
        self.time_attn = AxisMDTA(
            channels=channels,
            num_heads=num_heads,
            axis='time',
            bias=bias,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )
        
        # 用于自适应融合的投影
        self.fusion_norm = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T, F)
        
        Returns:
            out_freq: (B, C, T, F) 频率轴注意力输出
            out_time: (B, C, T, F) 时间轴注意力输出
        """
        # 并行处理两个轴
        out_freq = self.freq_attn(x)
        out_time = self.time_attn(x)
        
        return out_freq, out_time


# ================================
# [Legacy] Mamba 相关类（保留用于向后兼容）
# ================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Mamba 的核心
    
    实现选择性扫描机制:
        x_k = A_k * x_{k-1} + B_k * u_k
        y_k = C_k * x_k + D * u_k
    
    其中 A, B, C 是输入依赖的（选择性）
    
    数值稳定策略:
        1. A 使用负指数参数化: A = -exp(log_A) 确保离散化后稳定
        2. Delta (时间步长) 使用 softplus 确保正值
        3. 归一化状态更新防止梯度爆炸/消失
    
    Args:
        d_model: 输入通道数
        d_state: 状态空间维度 (N)
        d_conv: 局部卷积核大小
        expand: 内部扩展因子
        dt_min, dt_max: Delta 范围限制
        dt_init: Delta 初始化策略
        dt_scale: Delta 缩放因子
        bias: 是否使用偏置
        conv_bias: 卷积是否使用偏置
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # 输入投影: x -> (z, x_proj) 其中 z 用于门控
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 局部卷积（短程依赖捕获）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )
        
        # SSM 参数投影
        # B, C, Delta 从输入动态生成（选择性机制）
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # A 参数（对数空间，确保稳定性）
        # 初始化为 HiPPO 矩阵的近似
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner,
        )
        self.log_A = nn.Parameter(torch.log(A))
        
        # D 参数（直接通路）
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Delta 投影和初始化
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self._init_dt_proj(dt_init, dt_scale)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(self.d_inner)
    
    def _init_dt_proj(self, dt_init: str, dt_scale: float):
        """初始化 Delta 投影层，确保合理的时间步长范围"""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        # 逆 softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # 小权重初始化
        nn.init.normal_(self.dt_proj.weight, std=0.001 * dt_scale)
    
    def _ssm_scan(self, u: torch.Tensor, delta: torch.Tensor, 
                  A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        执行选择性状态空间扫描（序列方向）
        
        使用离散化公式:
            A_bar = exp(delta * A)
            B_bar = delta * B
            x_k = A_bar * x_{k-1} + B_bar * u_k
            y_k = C * x_k
        
        Args:
            u: (B, D, L) 输入序列
            delta: (B, D, L) 时间步长
            A: (D, N) 状态转移矩阵（对数空间）
            B: (B, N, L) 输入矩阵
            C: (B, N, L) 输出矩阵
        
        Returns:
            y: (B, D, L) 输出序列
        """
        B_batch, D, L = u.shape
        N = A.shape[1]
        
        # 离散化 A: A_bar = exp(delta * A)
        # delta: (B, D, L) -> (B, D, L, 1)
        # A: (D, N) -> (1, D, 1, N)
        delta_A = delta.unsqueeze(-1) * (-torch.exp(A)).unsqueeze(0).unsqueeze(2)
        A_bar = torch.exp(delta_A)  # (B, D, L, N)
        
        # 离散化 B: B_bar = delta * B
        # delta: (B, D, L) -> (B, D, L, 1)
        # B: (B, N, L) -> (B, 1, L, N)
        delta_B = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)  # (B, D, L, N)
        
        # 初始化状态
        x = torch.zeros(B_batch, D, N, device=u.device, dtype=u.dtype)
        
        # 输出容器
        ys = []
        
        # 序列扫描
        for t in range(L):
            # x_k = A_bar * x_{k-1} + B_bar * u_k
            x = A_bar[:, :, t, :] * x + delta_B[:, :, t, :] * u[:, :, t:t+1]
            # y_k = C * x_k
            # C[:, :, t]: (B, N) -> (B, 1, N)
            # x: (B, D, N)
            y_t = torch.einsum('bn,bdn->bd', C[:, :, t], x)
            ys.append(y_t)
        
        y = torch.stack(ys, dim=-1)  # (B, D, L)
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入序列
        
        Returns:
            y: (B, L, D) 输出序列
        """
        B, L, D = x.shape
        
        # 输入投影和分割
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)
        
        # 局部卷积
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # 裁剪到原始长度
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # SSM 参数投影
        ssm_params = self.x_proj(x_conv)  # (B, L, 2N+1)
        B_proj, C_proj, delta_raw = torch.split(
            ssm_params, [self.d_state, self.d_state, 1], dim=-1
        )
        
        # Delta: softplus 确保正值，并限制范围
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, d_inner)
        delta = torch.clamp(delta, min=self.dt_min, max=self.dt_max)
        delta = rearrange(delta, 'b l d -> b d l')
        
        # 重排列用于 SSM
        u = rearrange(x_conv, 'b l d -> b d l')
        B_proj = rearrange(B_proj, 'b l n -> b n l')
        C_proj = rearrange(C_proj, 'b l n -> b n l')
        
        # SSM 扫描
        y_ssm = self._ssm_scan(u, delta, self.log_A, B_proj, C_proj)
        
        # 直接通路
        y_ssm = y_ssm + u * self.D.unsqueeze(0).unsqueeze(-1)
        
        # 重排回 (B, L, D)
        y_ssm = rearrange(y_ssm, 'b d l -> b l d')
        y_ssm = self.norm(y_ssm)
        
        # 门控输出
        y = y_ssm * F.silu(z)
        
        # 输出投影
        y = self.out_proj(y)
        
        return y


class BidirectionalMamba(nn.Module):
    """
    双向 Mamba 模块
    
    结合正向和反向的选择性状态空间扫描，
    捕获序列的双向长程依赖。
    
    融合策略:
        1. 分别执行前向和后向 SSM
        2. 后向输出翻转对齐
        3. 使用可学习权重融合或简单相加
    
    数值稳定策略:
        1. 前后向分别归一化
        2. 融合后再次归一化
        3. 残差连接
    
    Args:
        d_model: 模型维度
        d_state: 状态空间维度
        d_conv: 局部卷积核大小
        expand: 扩展因子
        fusion: 融合方式 ('add', 'concat', 'gate')
        dropout: dropout 比例
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion: str = 'gate',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion = fusion
        
        # 前向 SSM
        self.ssm_forward = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 后向 SSM
        self.ssm_backward = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 融合层
        if fusion == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == 'gate':
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        
        # 归一化和 dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._sanity_checked = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入序列
        
        Returns:
            y: (B, L, D) 输出序列
        """
        identity = x
        
        # 前向扫描
        y_fwd = self.ssm_forward(x)
        
        # 后向扫描（翻转 -> SSM -> 翻转回来）
        x_rev = torch.flip(x, dims=[1])
        y_bwd_rev = self.ssm_backward(x_rev)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])
        
        # 融合
        if self.fusion == 'add':
            y = (y_fwd + y_bwd) / 2
        elif self.fusion == 'concat':
            y = self.fusion_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        elif self.fusion == 'gate':
            gate = self.fusion_gate(torch.cat([y_fwd, y_bwd], dim=-1))
            y = gate * y_fwd + (1 - gate) * y_bwd
        else:
            y = y_fwd + y_bwd
        
        # 残差 + 归一化
        y = self.dropout(y)
        y = self.norm(y + identity)
        
        # Sanity check (首次)
        if not self._sanity_checked and self.training:
            self._sanity_check(x, y_fwd, y_bwd)
            self._sanity_checked = True
        
        return y
    
    def _sanity_check(self, x, y_fwd, y_bwd):
        """验证双向扫描的有效性"""
        with torch.no_grad():
            # 检查前向和后向输出的相关性（应该不同但相关）
            fwd_flat = y_fwd.view(-1)
            bwd_flat = y_bwd.view(-1)
            
            if fwd_flat.std() > 1e-6 and bwd_flat.std() > 1e-6:
                corr = torch.corrcoef(torch.stack([fwd_flat, bwd_flat]))[0, 1]
                print(f"    [BiMamba] Forward-Backward correlation: {corr.item():.4f}")
                
                # 检查输出统计
                print(f"    [BiMamba] y_fwd: mean={y_fwd.mean():.4f}, std={y_fwd.std():.4f}")
                print(f"    [BiMamba] y_bwd: mean={y_bwd.mean():.4f}, std={y_bwd.std():.4f}")


class DepthwiseScan1D(nn.Module):
    """
    [Legacy] 简化的扫描模拟（使用 depthwise Conv1D + dilation）
    保留用于向后兼容，新代码请使用 BidirectionalMamba
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,  # depthwise
        )
        self.norm = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        out = self.norm(out)
        return F.gelu(out)


# ================================
# 1) 基础模块
# ================================

class ResidualBlock1D(nn.Module):
    """1D 残差块"""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.conv1(F.gelu(self.norm1(x)))
        out = self.dropout(out)
        out = self.conv2(F.gelu(self.norm2(out)))
        return out + identity


class DownBlock1D(nn.Module):
    """1D 下采样块"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock1D(out_ch, dropout)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
    
    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock1D(nn.Module):
    """1D 上采样块（带 skip connection）"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        # ConvTranspose 上采样
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        # Skip concat 后的 refinement
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.res = ResidualBlock1D(out_ch, dropout)
    
    def forward(self, x, skip):
        x = self.up(x)
        # 对齐长度（处理可能的 1 像素偏差）
        if x.shape[2] != skip.shape[2]:
            x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x


# ================================
# 2) 1D U-Net（时域主干）
# ================================

class UNet1D(nn.Module):
    """
    1D U-Net for time-domain reconstruction
    
    Args:
        in_ch: 输入通道数
        base_ch: 基础通道数
        levels: 编码器层数
        dropout: dropout 比例
    """
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        levels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.levels = levels
        
        # 输入投影
        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3)
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        ch = base_ch
        for i in range(levels):
            out_ch = ch * 2
            self.enc_blocks.append(DownBlock1D(ch, out_ch, dropout))
            ch = out_ch
        
        # Bottleneck
        self.bottleneck = ResidualBlock1D(ch, dropout)
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        for i in range(levels):
            skip_ch = ch // 2
            out_ch = ch // 2
            self.dec_blocks.append(UpBlock1D(ch, skip_ch, out_ch, dropout))
            ch = out_ch
        
        # 输出投影
        self.out_conv = nn.Conv1d(base_ch, in_ch, kernel_size=7, padding=3)
        
        # 用于 FiLM 调制的中间特征（decoder 的第 1 和第 2 层）
        self.film_layer_indices = [0, 1]  # 对应 dec_blocks 的索引
    
    def forward(self, x: torch.Tensor, film_params: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            x: (B, 1, L)
            film_params: 可选的 FiLM 参数字典
                {
                    "alpha_0": (B, C, L),
                    "beta_0": (B, C, L),
                    "alpha_1": (B, C, L),
                    "beta_1": (B, C, L),
                }
        
        Returns:
            y: (B, 1, L)
        """
        # 输入投影
        x = self.in_conv(x)
        
        # Encoder（保存 skip connections）
        skips = []
        h = x
        for i, enc in enumerate(self.enc_blocks):
            skips.append(h)
            h = enc(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder（逆序使用 skips）
        for i, dec in enumerate(self.dec_blocks):
            skip = skips[-(i+1)]
            h = dec(h, skip)
            
            # FiLM 调制（如果提供了参数）
            if film_params is not None and i in self.film_layer_indices:
                alpha_key = f"alpha_{i}"
                beta_key = f"beta_{i}"
                if alpha_key in film_params and beta_key in film_params:
                    alpha = film_params[alpha_key]
                    beta = film_params[beta_key]
                    # alpha, beta: (B, C, L_dec)
                    # h: (B, C, L_dec)
                    # 对齐长度
                    if alpha.shape[2] != h.shape[2]:
                        alpha = F.interpolate(alpha, size=h.shape[2], mode='linear', align_corners=False)
                        beta = F.interpolate(beta, size=h.shape[2], mode='linear', align_corners=False)
                    h = alpha * h + beta
        
        # 输出投影
        y = self.out_conv(h)
        
        return y


# ================================
# 3) 可替代的 Selective Scan 实现
# ================================

class DepthwiseScan1D(nn.Module):
    """
    简化的扫描模拟（使用 depthwise Conv1D + dilation）
    未来可替换为真实的 selective scan (e.g., Mamba)
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,  # depthwise
        )
        self.norm = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        out = self.norm(out)
        return F.gelu(out)


# ================================
# 4) ACSSBlock2D（核心模块 - MDTA 版本）
# ================================

class ACSSBlock2D(nn.Module):
    """
    Axis-Conditioned Selective Scan Block (2D) - MDTA 版本
    
    v3.0: 使用 Multi-Dconv Head Transposed Attention 替代 Mamba
    
    输入输出: (B, C, T, F)
    
    核心架构：
        1) Axis Summary: 提取频轴和时轴摘要
        2) Axis-conditioned Gate: 生成位置相关门控
        3) Dual-Axis MDTA: 并行的时间/频率注意力分支
        4) Adaptive Fusion + Residual + Norm
    
    设计优势：
        - 时间轴注意力 (T=33): 捕捉时间动态，全局感受野
        - 频率轴注意力 (F=101): 捕捉频率模式，跨频带交互
        - 自适应门控: 动态平衡两个分支的贡献
    """
    def __init__(
        self, 
        channels: int, 
        dropout: float = 0.0,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        self.channels = channels
        
        # 1) Axis Summary (mean+std pooling)
        # 频轴摘要: (B,C,T,F) -> (B,2C,T)
        # 时轴摘要: (B,C,T,F) -> (B,2C,F)
        self.summary_proj_freq = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.summary_proj_time = nn.Conv1d(channels * 2, channels, kernel_size=1)
        
        # 2) Gate 生成网络（基于频轴摘要）
        # 输出 (B, 1, T) 的门控权重
        self.gate_net = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 3) Dual-Axis MDTA（核心创新）
        # 频率轴注意力：捕捉跨频率 bin 的关系（如谐波、频带间耦合）
        self.freq_attn = AxisMDTA(
            channels=channels,
            num_heads=num_heads,
            axis='freq',
            bias=False,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )
        
        # 时间轴注意力：捕捉时间动态（如瞬态伪影、节律变化）
        self.time_attn = AxisMDTA(
            channels=channels,
            num_heads=num_heads,
            axis='time',
            bias=False,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )
        
        # 4) 输出投影和归一化
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T, F)
        
        Returns:
            out: (B, C, T, F)
            g_freq: (B, 1, T)  门控图（用于可视化和分析）
        """
        B, C, T, F = x.shape
        identity = x
        
        # 1) Axis Summary
        # 频轴摘要: 沿 F 维 pool
        x_mean_f = x.mean(dim=3)  # (B, C, T)
        x_std_f = x.std(dim=3)    # (B, C, T)
        s_f = torch.cat([x_mean_f, x_std_f], dim=1)  # (B, 2C, T)
        s_f = self.summary_proj_freq(s_f)  # (B, C, T)
        
        # 时轴摘要: 沿 T 维 pool
        x_mean_t = x.mean(dim=2)  # (B, C, F)
        x_std_t = x.std(dim=2)
        s_t = torch.cat([x_mean_t, x_std_t], dim=1)  # (B, 2C, F)
        s_t = self.summary_proj_time(s_t)  # (B, C, F)
        
        # 2) Gate 生成（基于频轴摘要）
        g_freq = self.gate_net(s_f)  # (B, 1, T)
        
        # 3) Dual-Axis MDTA（并行处理）
        U_freq = self.freq_attn(x)  # (B, C, T, F) - 频率全局性
        U_time = self.time_attn(x)  # (B, C, T, F) - 时间全局性
        
        # 4) Adaptive Fusion（自适应门控融合）
        # g_freq broadcast to (B, 1, T, 1)
        g = g_freq.unsqueeze(-1)  # (B, 1, T, 1)
        Y = g * U_freq + (1 - g) * U_time  # (B, C, T, F)
        
        # 5) 输出投影 + Residual + Norm
        out = self.proj(Y)
        out = self.dropout_layer(out)
        out = out + identity  # 残差连接
        out = self.norm(out)
        
        return out, g_freq


# ================================
# [Legacy] Mamba 扫描类（保留用于向后兼容）
# ================================

class ScanFreq(nn.Module):
    """
    [Legacy] 沿频率轴的双向 Mamba 扫描
    已被 AxisMDTA(axis='freq') 替代
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        # 使用 MDTA 替代
        self.scan = AxisMDTA(
            channels=channels,
            num_heads=4,
            axis='freq',
            dropout=0.0,
        )
        self._sanity_checked = False
    
    def forward(self, x):
        return self.scan(x)


class ScanTime(nn.Module):
    """
    [Legacy] 沿时间轴的双向 Mamba 扫描
    已被 AxisMDTA(axis='time') 替代
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        # 使用 MDTA 替代
        self.scan = AxisMDTA(
            channels=channels,
            num_heads=4,
            axis='time',
            dropout=0.0,
        )
        self._sanity_checked = False
    
    def forward(self, x):
        return self.scan(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T, F)
        
        Returns:
            out: (B, C, T, F)
            g_freq: (B, 1, T)  门控图
        """
        B, C, T, F = x.shape
        identity = x
        
        # 1) Axis Summary
        # 频轴摘要: 沿 F 维 pool
        x_mean_f = x.mean(dim=3)  # (B, C, T)
        x_std_f = x.std(dim=3)    # (B, C, T)
        s_f = torch.cat([x_mean_f, x_std_f], dim=1)  # (B, 2C, T)
        s_f = self.summary_proj_freq(s_f)  # (B, C, T)
        
        # 时轴摘要: 沿 T 维 pool
        x_mean_t = x.mean(dim=2)  # (B, C, F)
        x_std_t = x.std(dim=2)
        s_t = torch.cat([x_mean_t, x_std_t], dim=1)  # (B, 2C, F)
        s_t = self.summary_proj_time(s_t)  # (B, C, F)
        
        # 2) Gate（基于频轴摘要）
        g_freq = self.gate_net(s_f)  # (B, 1, T)
        
        # 3) Bidirectional Mamba Selective Scan
        U_freq = self.scan_freq(x)  # (B, C, T, F)
        U_time = self.scan_time(x)  # (B, C, T, F)
        
        # Mixture: g_freq broadcast to (B, 1, T, 1)
        g = g_freq.unsqueeze(-1)  # (B, 1, T, 1)
        Y = g * U_freq + (1 - g) * U_time  # (B, C, T, F)
        
        # 4) Residual + Norm
        out = self.proj(Y)
        out = self.dropout(out)
        out = out + identity
        out = self.norm(out)
        
        return out, g_freq


# ================================
# 5) SpecEncoder2D
# ================================

class SpecEncoder2D(nn.Module):
    """
    轻量 2D CNN，将谱图编码为特征
    输入: (B, F_sel, T)  log-magnitude spectrogram
    输出: (B, C, T, F_sel)
    """
    def __init__(self, in_freq: int, out_channels: int = 64, dropout: float = 0.0):
        super().__init__()
        # 输入投影: (B, 1, F, T) -> (B, C, F, T)
        self.in_conv = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        
        # 两层 2D 卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, F_sel, T)  log-mag spectrogram
        
        Returns:
            X: (B, C, T, F_sel)  注意维度顺序变化
        """
        # (B, F, T) -> (B, 1, F, T)
        x = S.unsqueeze(1)
        
        # Encode
        x = self.in_conv(x)   # (B, C, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 转换维度顺序: (B, C, F, T) -> (B, C, T, F)
        x = x.permute(0, 1, 3, 2)
        
        return x


# ================================
# 6) FiLM 调制模块
# ================================

class FiLMGenerator1D(nn.Module):
    """
    从时频特征生成 FiLM 参数（alpha, beta）
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # 为每个目标层生成 alpha 和 beta
        self.alpha_nets = nn.ModuleList()
        self.beta_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            self.alpha_nets.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(in_channels // 2, out_channels, kernel_size=1),
                )
            )
            self.beta_nets.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(in_channels // 2, out_channels, kernel_size=1),
                )
            )
        
        # 零初始化：确保训练初期 alpha≈1, beta≈0，不破坏主干
        self._init_film_params()
    
    def _init_film_params(self):
        """零初始化 FiLM 生成器，使初始状态为 H' = H"""
        for alpha_net in self.alpha_nets:
            # 最后一层权重为0，偏置为0（由于输出是 log_alpha，exp(0)=1）
            nn.init.zeros_(alpha_net[-1].weight)
            nn.init.zeros_(alpha_net[-1].bias)
        
        for beta_net in self.beta_nets:
            # 最后一层权重为0，偏置为0
            nn.init.zeros_(beta_net[-1].weight)
            nn.init.zeros_(beta_net[-1].bias)
    
    def forward(self, m: torch.Tensor, target_length: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            m: (B, C_m, T)  时频特征摘要
            target_length: 目标时域长度 L
        
        Returns:
            film_params: {
                "alpha_0": (B, C_out, L),
                "beta_0": (B, C_out, L),
                ...
            }
        """
        # 插值到目标长度
        m_L = F.interpolate(m, size=target_length, mode='linear', align_corners=False)
        
        film_params = {}
        for i in range(self.num_layers):
            alpha = self.alpha_nets[i](m_L)
            beta = self.beta_nets[i](m_L)
            film_params[f"alpha_{i}"] = alpha
            film_params[f"beta_{i}"] = beta
        
        return film_params


# ================================
# 7) 完整的 UAR-ACSSNet
# ================================

class UAR_ACSSNet(nn.Module):
    """
    Unified Artifact Removal with Axis-Conditioned Selective Scan Network
    
    v3.0: 使用 MDTA (Multi-Dconv Head Transposed Attention) 替代 Mamba
    
    完整端到端模型，包含:
        - 时域主干: 1D U-Net
        - 时频分支: SpecEncoder2D + ACSSStack2D (Dual-Axis MDTA)
        - 跨域融合: FiLM 调制
        - 置信图生成
    
    Args:
        segment_length: 输入信号长度（用于计算 STFT 时间帧）
        unet_base_ch: U-Net 基础通道数
        unet_levels: U-Net 编码器层数
        spec_channels: 谱图编码器输出通道数
        acss_depth: ACSSBlock 堆叠层数
        num_freq_bins: STFT 选定的频率 bin 数量（需与 STFTProcessor 一致）
        dropout: dropout 比例
        baseline_mode: 仅使用U-Net，不使用时频分支
        attn_num_heads: MDTA 注意力头数
        attn_ffn_expansion: MDTA FFN 扩展因子
    """
    def __init__(
        self,
        segment_length: int = 2048,
        unet_base_ch: int = 32,
        unet_levels: int = 4,
        spec_channels: int = 64,
        acss_depth: int = 3,
        num_freq_bins: int = 103,  # 1-100 Hz @ 500Hz, n_fft=512 -> 103 bins
        dropout: float = 0.0,
        baseline_mode: bool = False,  # 仅使用U-Net，不使用时频分支
        # MDTA 注意力参数
        attn_num_heads: int = 4,       # 注意力头数
        attn_ffn_expansion: float = 2.0,  # FFN 扩展因子
    ):
        super().__init__()
        self.segment_length = segment_length
        self.num_freq_bins = num_freq_bins
        self.baseline_mode = baseline_mode
        
        # 保存 MDTA 参数
        self.attn_num_heads = attn_num_heads
        self.attn_ffn_expansion = attn_ffn_expansion
        
        # STFT 处理器（固定参数）
        self.stft_proc = STFTProcessor(
            fs=500,
            n_fft=512,
            hop_length=64,
            win_length=156,
            freq_min=1.0,
            freq_max=100.0,
        )
        
        # 时域主干: 1D U-Net
        self.unet = UNet1D(
            in_ch=1,
            base_ch=unet_base_ch,
            levels=unet_levels,
            dropout=dropout,
        )
        
        # 仅在非baseline模式下初始化时频分支
        if not baseline_mode:
            # 时频分支: SpecEncoder2D
            self.spec_encoder = SpecEncoder2D(
                in_freq=num_freq_bins,
                out_channels=spec_channels,
                dropout=dropout,
            )
            
            # ACSSBlock 堆叠（使用 Dual-Axis MDTA）
            self.acss_blocks = nn.ModuleList([
                ACSSBlock2D(
                    channels=spec_channels, 
                    dropout=dropout,
                    num_heads=attn_num_heads,
                    ffn_expansion=attn_ffn_expansion,
                ) for _ in range(acss_depth)
            ])
            
            # 跨域融合: FiLM Generator
            # 从 (B, C, T, F) -> pool over F -> (B, C, T)
            self.film_pool = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, T, F) -> (B, C, T, 1)
            
            # FiLM 生成器（为 U-Net decoder 的前两层生成参数）
            # 目标通道数需要与 U-Net decoder 对应层匹配
            # decoder 第 0 层输出: unet_base_ch * 2^(levels-1)
            # decoder 第 1 层输出: unet_base_ch * 2^(levels-2)
            film_target_ch_0 = unet_base_ch * (2 ** (unet_levels - 1))
            film_target_ch_1 = unet_base_ch * (2 ** (unet_levels - 2))
            
            self.film_gen_0 = FiLMGenerator1D(spec_channels, film_target_ch_0, num_layers=1)
            self.film_gen_1 = FiLMGenerator1D(spec_channels, film_target_ch_1, num_layers=1)
            
            # 置信图生成（从 ACSS 输出）
            self.confidence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((None, 1)),  # (B, C, T, F) -> (B, C, T, 1)
                nn.Conv2d(spec_channels, spec_channels // 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(spec_channels // 2, 1, kernel_size=1),
            )
            # Zero initialization for confidence_head last layer
            nn.init.zeros_(self.confidence_head[-1].weight)
            nn.init.zeros_(self.confidence_head[-1].bias)
    
    def forward(self, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_raw: (B, 1, L)  输入时域信号
        
        Returns:
            {
                "y_hat": (B, 1, L)  重建信号
                "w": (B, 1, L)  置信图 [0, 1]
                "g_freq": (B, 1, T)  门控图（可选，用于可视化）
            }
        """
        B, C, L = x_raw.shape
        assert C == 1, f"Expected single channel, got {C}"
        assert L == self.segment_length, f"Expected length {self.segment_length}, got {L}"
        
        # ==================
        # Baseline模式：仅使用U-Net
        # ==================
        if self.baseline_mode:
            # 直接使用U-Net，不使用FiLM调制
            y_hat = self.unet(x_raw, film_params=None)  # (B, 1, L)
            
            # 返回简化输出（w固定为0.5，不参与训练）
            w = torch.ones_like(y_hat) * 0.5
            g_freq_avg = torch.zeros(B, 1, 1, device=x_raw.device)  # 占位符
            
            # 详细统计（仅首次）
            if not hasattr(self, '_output_checked'):
                print(f"\n  [Baseline Mode] Model Output Check:")
                print(f"    y_hat: shape={y_hat.shape}, range=[{y_hat.min():.3f}, {y_hat.max():.3f}]")
                print(f"    w: fixed at 0.5 (not trainable)")
                print(f"    ✓ Using pure U-Net without FiLM modulation")
                self._output_checked = True
            
            return {
                "y_hat": y_hat,
                "w": w,
                "g_freq": g_freq_avg,
            }
        
        # ==================
        # 完整模式：时频分支 + FiLM融合
        # ==================
        
        # STFT: (B, 1, L) -> (B, F_sel, T)
        S = self.stft_proc(x_raw)  # (B, F_sel, T)
        T_stft = S.shape[2]
        
        # SpecEncoder: (B, F_sel, T) -> (B, C_spec, T, F_sel)
        X_tf = self.spec_encoder(S)  # (B, C_spec, T, F)
        
        # ACSSBlock 堆叠
        g_freq_list = []
        for acss_block in self.acss_blocks:
            X_tf, g_freq = acss_block(X_tf)
            g_freq_list.append(g_freq)
        
        # 平均门控图（用于可视化）
        g_freq_avg = torch.stack(g_freq_list, dim=0).mean(dim=0)  # (B, 1, T)
        
        # ==================
        # 跨域融合（FiLM）
        # ==================
        
        # 从 X_tf 提取时间维摘要: (B, C, T, F) -> pool F -> (B, C, T)
        m = self.film_pool(X_tf).squeeze(-1)  # (B, C, T)
        
        # 为 U-Net 不同层生成 FiLM 参数
        # 需要先计算每层的时域长度
        # U-Net 的 decoder 层长度递增：L / 2^(levels-i)
        L_dec_0 = L // (2 ** (self.unet.levels - 1))
        L_dec_1 = L // (2 ** (self.unet.levels - 2))
        
        film_params_0 = self.film_gen_0(m, L_dec_0)
        film_params_1 = self.film_gen_1(m, L_dec_1)
        
        # 合并参数
        film_params = {
            **{k.replace("_0", "_0"): v for k, v in film_params_0.items()},
            **{k.replace("_0", "_1"): v for k, v in film_params_1.items()},
        }
        
        # ==================
        # 时域主干（带 FiLM 调制）
        # ==================
        
        y_hat = self.unet(x_raw, film_params)  # (B, 1, L)
        
        # ==================
        # 置信图生成
        # ==================
        
        w_tf = self.confidence_head(X_tf)  # (B, 1, T, 1)
        w_t = w_tf.squeeze(-1)  # (B, 1, T)
        
        # 插值到时域长度
        w = F.interpolate(w_t, size=L, mode='linear', align_corners=False)  # (B, 1, L)
        w = torch.sigmoid(w)  # 限制到 [0, 1]
        
        # ==================
        # 输出检查
        # ==================
        
        assert y_hat.shape == (B, 1, L), f"y_hat shape mismatch: {y_hat.shape}"
        assert w.shape == (B, 1, L), f"w shape mismatch: {w.shape}"
        assert (w >= 0).all() and (w <= 1).all(), f"w out of range: [{w.min()}, {w.max()}]"
        
        # 详细统计（仅首次）
        if not hasattr(self, '_output_checked'):
            print(f"\n  Model Output Check:")
            print(f"    y_hat: shape={y_hat.shape}, range=[{y_hat.min():.3f}, {y_hat.max():.3f}]")
            print(f"    w: shape={w.shape}, range=[{w.min():.3f}, {w.max():.3f}], mean={w.mean():.3f}")
            print(f"    g_freq: shape={g_freq_avg.shape}, range=[{g_freq_avg.min():.3f}, {g_freq_avg.max():.3f}]")
            
            # 检查 w 是否退化
            if w.mean() > 0.45 and w.mean() < 0.55:
                print(f"    ⚠ Warning: w.mean()={w.mean():.3f} close to 0.5 (may indicate saturation)")
            if w.std() < 0.01:
                print(f"    ⚠ Warning: w.std()={w.std():.4f} very small (may indicate collapse)")
            else:
                print(f"    ✓ w statistics look healthy (std={w.std():.4f})")
            
            self._output_checked = True
        
        return {
            "y_hat": y_hat,
            "w": w,
            "g_freq": g_freq_avg,  # (B, 1, T)
        }


def test_model():
    """测试模型前向传播（MDTA 版本）"""
    print("\n=== Testing UAR-ACSSNet v3.0 (Dual-Axis MDTA) ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 测试配置
    print("\n[1] Model Configuration:")
    print(f"    - Dual-Axis MDTA (Multi-Dconv Head Transposed Attention)")
    print(f"    - num_heads=4")
    print(f"    - ffn_expansion=2.0")
    print(f"    - Parallel time/freq attention branches")
    print(f"    - Adaptive gating fusion")
    
    model = UAR_ACSSNet(
        segment_length=2048,
        unet_base_ch=32,
        unet_levels=4,
        spec_channels=64,
        acss_depth=3,
        num_freq_bins=103,
        dropout=0.1,
        # MDTA 参数
        attn_num_heads=4,
        attn_ffn_expansion=2.0,
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] Model Parameters:")
    print(f"    Total params: {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    
    # 统计各模块参数
    unet_params = sum(p.numel() for p in model.unet.parameters())
    if hasattr(model, 'acss_blocks'):
        acss_params = sum(p.numel() for p in model.acss_blocks.parameters())
        spec_params = sum(p.numel() for p in model.spec_encoder.parameters())
        print(f"    U-Net: {unet_params:,}")
        print(f"    SpecEncoder: {spec_params:,}")
        print(f"    ACSS (MDTA): {acss_params:,}")
    
    # 前向传播测试
    print(f"\n[3] Forward Pass Test:")
    x = torch.randn(4, 1, 2048).to(device)
    
    # 训练模式（触发 sanity check）
    model.train()
    print(f"\n[MDTA Sanity Checks - first forward in training mode]")
    outputs = model(x)
    
    # 评估模式
    model.eval()
    print(f"\n[4] Inference Test:")
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutputs:")
    print(f"  Type: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"  Keys: {list(outputs.keys())}")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
    
    # 验证 shape 和范围
    assert outputs["y_hat"].shape == (4, 1, 2048), "y_hat shape mismatch"
    assert outputs["w"].shape == (4, 1, 2048), "w shape mismatch"
    assert (outputs["w"] >= 0).all() and (outputs["w"] <= 1).all(), "w out of range"
    
    # 详细检查 w 的统计特性
    w = outputs["w"]
    print(f"\n[5] Confidence Map (w) Statistics:")
    print(f"    min: {w.min():.6f}")
    print(f"    max: {w.max():.6f}")
    print(f"    mean: {w.mean():.6f}")
    print(f"    std: {w.std():.6f}")
    
    # 检查是否退化
    if 0.49 < w.mean() < 0.51:
        print(f"    ⚠ w.mean() very close to 0.5 (saturation check)")
    if w.std() < 0.01:
        print(f"    ⚠ w.std() very small (collapse check)")
    if w.min() > 0.4 and w.max() < 0.6:
        print(f"    ⚠ w range very narrow (limited dynamics)")
    
    # 测试梯度流
    print(f"\n[6] Gradient Flow Test:")
    model.train()
    x.requires_grad = True
    outputs = model(x)
    loss = outputs["y_hat"].sum()
    loss.backward()
    
    # 检查梯度
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    max_grad = max(grad_norms.values()) if grad_norms else 0
    min_grad = min(grad_norms.values()) if grad_norms else 0
    print(f"    Gradient norm range: [{min_grad:.6f}, {max_grad:.6f}]")
    
    # 检查 Attention 模块的梯度
    attn_grads = {k: v for k, v in grad_norms.items() if 'attn' in k.lower() or 'qkv' in k.lower()}
    if attn_grads:
        print(f"    Attention-related gradients: {len(attn_grads)} params")
        print(f"    Attention grad range: [{min(attn_grads.values()):.6f}, {max(attn_grads.values()):.6f}]")
    
    print("\n✓ All tests passed!")
    print("✓ Dual-Axis MDTA integration successful!\n")


def test_mdta_components():
    """单独测试 MDTA 组件"""
    print("\n=== Testing MDTA Components ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试 MDTA (2D)
    print("[1] Testing MDTA (2D Transposed Attention):")
    mdta = MDTA(channels=64, num_heads=4).to(device)
    x_2d = torch.randn(2, 64, 33, 101).to(device)  # (B, C, T, F)
    y_2d = mdta(x_2d)
    print(f"    Input: {x_2d.shape}")
    print(f"    Output: {y_2d.shape}")
    assert y_2d.shape == x_2d.shape, "Shape mismatch in MDTA"
    print(f"    ✓ MDTA passed")
    
    # 测试 AxisMDTA (freq)
    print("\n[2] Testing AxisMDTA (freq axis):")
    freq_attn = AxisMDTA(channels=64, num_heads=4, axis='freq').to(device)
    freq_attn.train()  # 触发 sanity check
    y_freq = freq_attn(x_2d)
    print(f"    Input: {x_2d.shape}")
    print(f"    Output: {y_freq.shape}")
    assert y_freq.shape == x_2d.shape, "Shape mismatch in AxisMDTA (freq)"
    print(f"    ✓ AxisMDTA (freq) passed")
    
    # 测试 AxisMDTA (time)
    print("\n[3] Testing AxisMDTA (time axis):")
    time_attn = AxisMDTA(channels=64, num_heads=4, axis='time').to(device)
    time_attn.train()
    y_time = time_attn(x_2d)
    print(f"    Input: {x_2d.shape}")
    print(f"    Output: {y_time.shape}")
    assert y_time.shape == x_2d.shape, "Shape mismatch in AxisMDTA (time)"
    print(f"    ✓ AxisMDTA (time) passed")
    
    # 测试 DualAxisMDTA
    print("\n[4] Testing DualAxisMDTA:")
    dual_attn = DualAxisMDTA(channels=64, num_heads=4).to(device)
    dual_attn.train()
    y_freq, y_time = dual_attn(x_2d)
    print(f"    Input: {x_2d.shape}")
    print(f"    Output freq: {y_freq.shape}")
    print(f"    Output time: {y_time.shape}")
    assert y_freq.shape == x_2d.shape, "Shape mismatch in DualAxisMDTA (freq)"
    assert y_time.shape == x_2d.shape, "Shape mismatch in DualAxisMDTA (time)"
    print(f"    ✓ DualAxisMDTA passed")
    
    # 测试 ACSSBlock2D (MDTA 版本)
    print("\n[5] Testing ACSSBlock2D (MDTA):")
    acss = ACSSBlock2D(channels=64, num_heads=4, ffn_expansion=2.0).to(device)
    acss.train()
    y_acss, g = acss(x_2d)
    print(f"    Input: {x_2d.shape}")
    print(f"    Output: {y_acss.shape}")
    print(f"    Gate: {g.shape}")
    assert y_acss.shape == x_2d.shape, "Shape mismatch in ACSSBlock2D"
    print(f"    ✓ ACSSBlock2D passed")
    
    # 测试计算效率对比
    print("\n[6] Computational Efficiency Analysis:")
    T, F, C = 33, 101, 64
    print(f"    Input dimensions: T={T}, F={F}, C={C}")
    
    # 标准注意力复杂度
    std_attn_complexity = (T * F) ** 2
    print(f"    Standard Attention O((T×F)²) = {std_attn_complexity:,}")
    
    # 转置注意力复杂度
    transposed_complexity = C ** 2
    print(f"    Transposed Attention O(C²) = {transposed_complexity:,}")
    
    # 轴向注意力复杂度
    axial_complexity = T ** 2 + F ** 2
    print(f"    Axial Attention O(T²+F²) = {axial_complexity:,}")
    
    speedup = std_attn_complexity / axial_complexity
    print(f"    Axial Speedup: {speedup:.1f}x faster than standard")
    
    print("\n✓ All MDTA component tests passed!\n")


if __name__ == "__main__":
    test_mdta_components()
    test_model()
