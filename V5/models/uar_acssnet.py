"""
UAR-ACSSNet: Unified Artifact Removal with Axis-Conditioned Selective Scan Network
单通道 EEG 去伪影端到端模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import sys
from pathlib import Path

# 确保可以导入 signal_processing
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from signal_processing import STFTProcessor


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


class ScanFreq(nn.Module):
    """沿频率轴的扫描聚合（针对 (B,C,T,F) 输入）"""
    def __init__(self, channels: int):
        super().__init__()
        # 对每个 (t) 位置，沿 f 维度扫描
        # 实现：permute 到 (B,C,F,T) 后做 depthwise conv
        # 增大 kernel_size 到 31 以捕捉频域长程依赖 (101 bins)
        self.scan = DepthwiseScan1D(channels, kernel_size=31)
        self._sanity_checked = False
    
    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        # (B, C, T, F) -> (B*T, C, F)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B*T, C, F)
        out = self.scan(x_reshaped)  # (B*T, C, F)
        out = out.reshape(B, T, C, F).permute(0, 2, 1, 3)  # -> (B, C, T, F)
        
        # Sanity check: 脉冲输入测试（仅首次）
        if not self._sanity_checked and self.training:
            self._sanity_check_pulse(x)
            self._sanity_checked = True
        
        return out
    
    def _sanity_check_pulse(self, x):
        """脉冲输入 sanity check：验证频率轴扫描方向"""
        with torch.no_grad():
            B, C, T, F = x.shape
            # 创建脉冲：在中心频率处设置脉冲
            pulse = torch.zeros(1, C, 1, F, device=x.device)
            pulse[:, :, :, F//2] = 1.0
            
            # 前向传播
            pulse_reshaped = pulse.permute(0, 2, 1, 3).reshape(1, C, F)
            out = self.scan(pulse_reshaped)
            out_reshaped = out.reshape(1, 1, C, F).permute(0, 2, 1, 3).squeeze(2)  # (1, C, F)
            
            # 检查扩散
            nonzero_indices = (out_reshaped[0, 0, :] > 1e-3).nonzero(as_tuple=True)[0]
            if len(nonzero_indices) > 1:
                spread = nonzero_indices.max() - nonzero_indices.min() + 1
                print(f"    [ScanFreq] Pulse spread along F axis: {spread.item()} bins (kernel effective)")
            else:
                print(f"    [ScanFreq] No spreading detected (identity-like)")


class ScanTime(nn.Module):
    """沿时间轴的扫描聚合（针对 (B,C,T,F) 输入）"""
    def __init__(self, channels: int):
        super().__init__()
        # 对每个 (f) 位置，沿 t 维度扫描
        self.scan = DepthwiseScan1D(channels, kernel_size=5)
        self._sanity_checked = False
    
    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        # (B, C, T, F) -> (B*F, C, T)
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B*F, C, T)
        out = self.scan(x_reshaped)  # (B*F, C, T)
        out = out.reshape(B, F, C, T).permute(0, 2, 3, 1)  # -> (B, C, T, F)
        
        # Sanity check: 脉冲输入测试（仅首次）
        if not self._sanity_checked and self.training:
            self._sanity_check_pulse(x)
            self._sanity_checked = True
        
        return out
    
    def _sanity_check_pulse(self, x):
        """脉冲输入 sanity check：验证时间轴扫描方向"""
        with torch.no_grad():
            B, C, T, F = x.shape
            # 创建脉冲：在中心时间处设置脉冲
            pulse = torch.zeros(1, C, T, 1, device=x.device)
            pulse[:, :, T//2, :] = 1.0
            
            # 前向传播
            pulse_reshaped = pulse.permute(0, 3, 1, 2).reshape(1, C, T)
            out = self.scan(pulse_reshaped)
            out_reshaped = out.reshape(1, 1, C, T).permute(0, 2, 3, 1).squeeze(3)  # (1, C, T)
            
            # 检查扩散
            nonzero_indices = (out_reshaped[0, 0, :] > 1e-3).nonzero(as_tuple=True)[0]
            if len(nonzero_indices) > 1:
                spread = nonzero_indices.max() - nonzero_indices.min() + 1
                print(f"    [ScanTime] Pulse spread along T axis: {spread.item()} frames (kernel effective)")
            else:
                print(f"    [ScanTime] No spreading detected (identity-like)")


# ================================
# 4) ACSSBlock2D（核心模块）
# ================================

class ACSSBlock2D(nn.Module):
    """
    Axis-Conditioned Selective Scan Block (2D)
    
    输入输出: (B, C, T, F)
    
    包含:
        1) Axis Summary: 提取频轴和时轴摘要
        2) Axis-conditioned Gate: 生成位置相关门控
        3) Selective Scan Mixture: 扫描聚合并混合
        4) Residual + Norm
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        
        # 1) Axis Summary (mean+std pooling)
        # 频轴摘要: (B,C,T,F) -> (B,2C,T)
        # 时轴摘要: (B,C,T,F) -> (B,2C,F)
        self.summary_proj_freq = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.summary_proj_time = nn.Conv1d(channels * 2, channels, kernel_size=1)
        
        # 2) Gate 生成网络（基于频轴摘要）
        self.gate_net = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 3) Selective Scan（可替代接口）
        self.scan_freq = ScanFreq(channels)
        self.scan_time = ScanTime(channels)
        
        # 4) 输出投影
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
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
        
        # 3) Selective Scan
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
    
    完整端到端模型，包含:
        - 时域主干: 1D U-Net
        - 时频分支: SpecEncoder2D + ACSSStack2D
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
    ):
        super().__init__()
        self.segment_length = segment_length
        self.num_freq_bins = num_freq_bins
        
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
        
        # 时频分支: SpecEncoder2D
        self.spec_encoder = SpecEncoder2D(
            in_freq=num_freq_bins,
            out_channels=spec_channels,
            dropout=dropout,
        )
        
        # ACSSBlock 堆叠
        self.acss_blocks = nn.ModuleList([
            ACSSBlock2D(spec_channels, dropout) for _ in range(acss_depth)
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
        # 时频分支
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
    """测试模型前向传播"""
    print("\n=== Testing UAR-ACSSNet ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UAR_ACSSNet(
        segment_length=2048,
        unet_base_ch=32,
        unet_levels=4,
        spec_channels=64,
        acss_depth=3,
        num_freq_bins=103,
        dropout=0.1,
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")
    
    # 前向传播测试
    x = torch.randn(4, 1, 2048).to(device)
    
    print(f"\n[Scan Sanity Check - will trigger during first forward]")
    
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutputs:")
    print(f"  Type: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"  Keys: {list(outputs.keys())}")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
    else:
        print(f"  Warning: Expected dict, got {type(outputs)}")
    
    # 验证 shape 和范围
    assert outputs["y_hat"].shape == (4, 1, 2048)
    assert outputs["w"].shape == (4, 1, 2048)
    assert (outputs["w"] >= 0).all() and (outputs["w"] <= 1).all()
    
    # 详细检查 w 的统计特性
    w = outputs["w"]
    print(f"\n  Confidence Map (w) Statistics:")
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
    
    print("\n✓ All tests passed!\n")


if __name__ == "__main__":
    test_model()
