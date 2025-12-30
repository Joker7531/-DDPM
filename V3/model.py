"""
Residual Noise Prediction U-Net for STFT-domain EEG Denoising (V3)

Key differences from V2:
1. Predicts noise residual (Raw - Clean) instead of clean signal
2. Log + InstanceNorm preprocessing
3. base_channels=32 for memory efficiency

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LogInstanceNorm(nn.Module):
    """
    Log + Instance Normalization preprocessing layer.
    
    Applies log(1 + |x|) * sign(x) transformation followed by Instance Norm.
    This helps stabilize training and handle the wide dynamic range of STFT.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log compression: log(1 + |x|) * sign(x)
        x_log = torch.sign(x) * torch.log1p(torch.abs(x))
        # Instance normalization
        x_norm = self.instance_norm(x_log)
        return x_norm


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of intermediate channels
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block with MaxPool followed by DoubleConv.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block with Upsample + Concatenation + DoubleConv.
    
    Args:
        in_channels (int): Number of input channels (upsampled + skip)
        out_channels (int): Number of output channels
        bilinear (bool): Use bilinear upsampling if True, else transposed conv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # For transposed conv, we need to upsample the decoder feature map
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2,  # Only upsample the decoder part
                kernel_size=2, stride=2
            )
        
        # After concatenation, we have in_channels, output to out_channels
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from decoder (lower resolution)
            x2: Skip connection from encoder (higher resolution)
        """
        x1 = self.up(x1)
        
        # Pad x1 to match x2 dimensions if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResidualNoiseUNet(nn.Module):
    """
    U-Net for predicting noise residual in STFT domain.
    
    The model predicts noise = Raw - Clean, and during inference:
    Clean_Pred = Raw - Model(Raw)
    
    Features:
    - Log + InstanceNorm preprocessing
    - Residual (noise) prediction paradigm
    - Memory-efficient base_channels=32
    
    Args:
        in_channels (int): Number of input channels (2 for Real/Imag)
        out_channels (int): Number of output channels (2 for Real/Imag)
        base_channels (int): Base number of channels (default: 32)
        depth (int): Depth of U-Net (number of downsampling steps)
        bilinear (bool): Use bilinear upsampling if True
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear
        
        # Log + InstanceNorm preprocessing
        self.preprocess = LogInstanceNorm(in_channels)
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Build channel list for encoder
        # e.g., depth=4, base=32: [32, 64, 128, 256, 512]
        self.encoder_channels = [base_channels]
        for i in range(depth):
            self.encoder_channels.append(self.encoder_channels[-1] * 2)
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(Down(self.encoder_channels[i], self.encoder_channels[i+1]))
        
        # Decoder path
        # Each decoder takes concatenated (upsampled + skip) and outputs to next level
        self.decoders = nn.ModuleList()
        for i in range(depth):
            # Input: current decoder channels + skip channels
            dec_in = self.encoder_channels[depth - i]  # From bottleneck upward
            skip_ch = self.encoder_channels[depth - i - 1]  # Skip from encoder
            dec_out = skip_ch  # Output matches skip level
            self.decoders.append(Up(dec_in + skip_ch, dec_out, bilinear))
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Print architecture summary
        print(f"  Encoder channels: {self.encoder_channels}")
        print(f"  Decoder: {self.encoder_channels[-1]} -> {base_channels}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise residual.
        
        Args:
            x: Input STFT [B, 2, F, T] (Raw noisy signal)
            
        Returns:
            Predicted noise [B, 2, F, T]
        """
        # Save original dimensions
        orig_h, orig_w = x.shape[2], x.shape[3]
        
        # Pad to multiple of 2^depth for U-Net compatibility
        factor = 2 ** self.depth
        pad_h = (factor - orig_h % factor) % factor
        pad_w = (factor - orig_w % factor) % factor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        
        # Preprocessing: Log + InstanceNorm
        x = self.preprocess(x)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Encoder path - store skip connections
        skips = [x1]
        h = x1
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
        
        # Remove last (bottleneck) from skips as it's already in h
        skips = skips[:-1]
        
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i+1)]  # Get skip connection from encoder
            h = decoder(h, skip)
        
        # Output
        noise = self.outc(h)
        
        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            noise = noise[:, :, :orig_h, :orig_w]
        
        return noise
    
    def denoise(self, raw_stft: torch.Tensor) -> torch.Tensor:
        """
        Denoise by predicting and subtracting noise.
        
        Clean = Raw - Predicted_Noise
        
        Args:
            raw_stft: Raw noisy STFT [B, 2, F, T]
            
        Returns:
            Denoised STFT [B, 2, F, T]
        """
        predicted_noise = self.forward(raw_stft)
        clean_pred = raw_stft - predicted_noise
        return clean_pred


def test_model():
    """Test the ResidualNoiseUNet model."""
    print("=" * 70)
    print("Testing ResidualNoiseUNet (V3)")
    print("=" * 70)
    
    # Test configuration
    batch_size = 4
    in_channels = 2
    freq_bins = 257  # STFT frequency bins (n_fft=512)
    time_frames = 625  # ~20 seconds
    base_channels = 32
    depth = 4
    
    # Create model
    model = ResidualNoiseUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        depth=depth
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Configuration:")
    print(f"  Input channels:  {in_channels}")
    print(f"  Base channels:   {base_channels}")
    print(f"  Depth:           {depth}")
    print(f"  Total params:    {n_params:,}")
    print(f"  Trainable:       {n_trainable:,}")
    print(f"  Model size:      {n_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, freq_bins, time_frames)
    print(f"\nInput shape:  {list(x.shape)}")
    
    # Forward pass (predict noise)
    model.eval()
    with torch.no_grad():
        noise_pred = model(x)
        clean_pred = model.denoise(x)
    
    print(f"Noise shape:  {list(noise_pred.shape)}")
    print(f"Clean shape:  {list(clean_pred.shape)}")
    
    # Verify residual relationship
    verify_clean = x - noise_pred
    assert torch.allclose(verify_clean, clean_pred, atol=1e-6), "Residual verification failed!"
    print("\n✓ Residual verification passed: Clean = Raw - Noise")
    
    # Test with odd dimensions
    x_odd = torch.randn(2, 2, 103, 450)
    with torch.no_grad():
        noise_odd = model(x_odd)
    
    assert noise_odd.shape == x_odd.shape, f"Shape mismatch: {noise_odd.shape} vs {x_odd.shape}"
    print(f"✓ Odd dimension test passed: {list(x_odd.shape)} -> {list(noise_odd.shape)}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
