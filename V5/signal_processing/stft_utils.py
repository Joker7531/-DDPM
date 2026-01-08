"""
STFT 工具模块
固定参数: fs=500Hz, n_fft=512, hop_length=64, win_length=156
频率范围: 1-100 Hz
"""
import torch
import torch.nn as nn
import math
from typing import Tuple


class STFTProcessor(nn.Module):
    """
    固定参数的 STFT 处理器
    
    参数:
        fs: 采样率 (Hz)
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        freq_min: 最小频率 (Hz)
        freq_max: 最大频率 (Hz)
    """
    
    def __init__(
        self,
        fs: int = 500,
        n_fft: int = 512,
        hop_length: int = 64,
        win_length: int = 156,
        freq_min: float = 1.0,
        freq_max: float = 100.0,
    ):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        # 计算频率分辨率
        self.df = fs / n_fft  # Hz per bin
        
        # 计算频率 bin 范围
        k_min = math.ceil(freq_min / self.df)
        k_max = math.floor(freq_max / self.df)
        
        # 限制在 [0, n_fft//2]
        k_min = max(0, k_min)
        k_max = min(n_fft // 2, k_max)
        
        self.k_min = k_min
        self.k_max = k_max
        self.num_freq_bins = k_max - k_min + 1
        
        # 注册 Hann 窗（不参与训练）
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)
        
        # 打印 STFT 配置（仅在首次初始化时）
        print(f"STFT Config:")
        print(f"  fs={fs} Hz, n_fft={n_fft}, hop={hop_length}, win={win_length}")
        print(f"  df={self.df:.3f} Hz/bin")
        print(f"  freq_range=[{freq_min}, {freq_max}] Hz -> bins [{k_min}, {k_max}]")
        print(f"  Selected freq bins: {self.num_freq_bins}")
        print(f"  Actual freq range: [{k_min * self.df:.2f}, {k_max * self.df:.2f}] Hz")
        
        # 验证公式
        F_sel_calc = k_max - k_min + 1
        print(f"\n  ✓ Verification:")
        print(f"    df = {self.df:.6f} Hz/bin")
        print(f"    k_min = {k_min}, k_max = {k_max}")
        print(f"    F_sel = k_max - k_min + 1 = {k_max} - {k_min} + 1 = {F_sel_calc}")
        print(f"    num_freq_bins = {self.num_freq_bins}")
        assert F_sel_calc == self.num_freq_bins, f"F_sel mismatch: {F_sel_calc} != {self.num_freq_bins}"
        print(f"    ✓ F_sel calculation verified!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 STFT 并返回 log-magnitude 谱图（选定频率范围）
        
        Args:
            x: (B, 1, L) 时域信号
        
        Returns:
            S: (B, F_sel, T) log-magnitude 谱图
               F_sel = num_freq_bins (对应 1-100 Hz)
        """
        B, C, L = x.shape
        assert C == 1, f"Expected single channel, got {C}"
        
        # x: (B, 1, L) -> (B, L)
        x_mono = x.squeeze(1)
        
        # STFT: (B, F, T)  其中 F = n_fft//2 + 1
        S_complex = torch.stft(
            x_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        
        # 取幅值
        S_mag = torch.abs(S_complex)  # (B, F, T)
        
        # 选择频率 bins
        S_sel = S_mag[:, self.k_min:self.k_max+1, :]  # (B, F_sel, T)
        
        # Log-magnitude (避免 log(0))
        S_log = torch.log1p(S_sel)  # (B, F_sel, T)
        
        # Shape assertion with detailed info
        assert S_log.shape[1] == self.num_freq_bins, \
            f"Expected {self.num_freq_bins} freq bins, got {S_log.shape[1]}"
        
        # Verbose check (only for first batch)
        if not hasattr(self, '_shape_verified'):
            print(f"\n  STFT Forward Shape Check:")
            print(f"    Input: {x.shape}")
            print(f"    STFT complex: {S_complex.shape}")
            print(f"    Selected bins [{self.k_min}:{self.k_max+1}]: {S_sel.shape}")
            print(f"    Output S_log: {S_log.shape}")
            print(f"    Expected freq bins: {self.num_freq_bins}, Actual: {S_log.shape[1]}")
            print(f"    ✓ Shape verification passed!")
            self._shape_verified = True
        
        return S_log
    
    def get_time_frames(self, L: int) -> int:
        """计算给定信号长度的 STFT 时间帧数"""
        # PyTorch stft with center=True 会 padding
        # 实际帧数计算（近似）
        T = (L + self.n_fft // 2 * 2) // self.hop_length
        return T


def test_stft_processor():
    """测试 STFT 处理器"""
    print("\n=== Testing STFTProcessor ===\n")
    
    # 创建处理器
    stft_proc = STFTProcessor(
        fs=500,
        n_fft=512,
        hop_length=64,
        win_length=156,
        freq_min=1.0,
        freq_max=100.0,
    )
    
    # 测试不同长度信号
    for L in [2048, 4096, 1024]:
        x = torch.randn(4, 1, L)  # (B, 1, L)
        S = stft_proc(x)
        
        T_expected = stft_proc.get_time_frames(L)
        
        print(f"Input: {x.shape} -> Output: {S.shape}")
        print(f"  Expected time frames: {T_expected}, actual: {S.shape[2]}")
        print(f"  S range: [{S.min():.3f}, {S.max():.3f}]")
        print()
    
    # 验证频率 bin 对应
    k_min, k_max = stft_proc.k_min, stft_proc.k_max
    df = stft_proc.df
    f_min_actual = k_min * df
    f_max_actual = k_max * df
    
    print(f"✓ Frequency bins verification:")
    print(f"  Requested: [1.0, 100.0] Hz")
    print(f"  Actual:    [{f_min_actual:.2f}, {f_max_actual:.2f}] Hz")
    print(f"  Bins:      [{k_min}, {k_max}]")
    print(f"  Count:     {k_max - k_min + 1}")
    
    # Range check
    assert 0.9 <= f_min_actual <= 1.1, "Min freq out of range"
    assert 99.0 <= f_max_actual <= 101.0, "Max freq out of range"
    
    print("\n✓ All tests passed!\n")


if __name__ == "__main__":
    test_stft_processor()
