"""
SpectrogramUNet: 2D U-Net for STFT-domain EEG Signal Reconstruction

This module implements a production-ready U-Net architecture specifically designed
for processing complex-valued STFT spectrograms represented as dual-channel (Real/Imag)
tensors with odd/prime frequency dimensions.

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm2d -> LeakyReLU) x 2
    
    This is the fundamental building block used throughout the U-Net architecture.
    Each block performs two sequential convolutions with batch normalization and
    LeakyReLU activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        negative_slope (float): Negative slope for LeakyReLU activation (default: 0.1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through double convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W]
        """
        return self.double_conv(x)


class SpectrogramUNet(nn.Module):
    """
    2D U-Net for STFT Spectrogram Reconstruction
    
    This U-Net architecture is specifically designed to handle STFT spectrograms
    with odd/prime frequency dimensions (e.g., 103) that don't align with standard
    pooling operations. It implements automatic padding and cropping to ensure
    dimension consistency.
    
    Architecture:
        - Encoder: 4 levels of DoubleConv -> MaxPool2d
        - Bottleneck: Single DoubleConv at lowest resolution
        - Decoder: 4 levels of TransposedConv2d -> Skip Connection -> DoubleConv
        - Output: 1x1 Conv to map back to 2 channels (Real/Imag)
    
    Input Shape: [Batch_Size, 2, 103, Time]
        - Channel 0: Real part of STFT
        - Channel 1: Imaginary part of STFT
    
    Output Shape: [Batch_Size, 2, 103, Time] (same as input)
    
    Args:
        in_channels (int): Number of input channels (default: 2 for Real/Imag)
        out_channels (int): Number of output channels (default: 2 for Real/Imag)
        base_channels (int): Base number of channels in first layer (default: 64)
        depth (int): Depth of the U-Net (default: 4)
    """
    
    def __init__(
        self, 
        in_channels: int = 2, 
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 4
    ):
        super(SpectrogramUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Encoder path (Downsampling)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = base_channels
        for i in range(depth):
            in_ch = in_channels if i == 0 else channels // 2
            self.encoders.append(DoubleConv(in_ch, channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = DoubleConv(channels // 2, channels)
        
        # Decoder path (Upsampling)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            self.upconvs.append(
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(channels, channels // 2))
            channels //= 2
        
        # Output layer (1x1 convolution, no activation)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def _calculate_padding(self, height: int, width: int, divisor: int = 16) -> Tuple[int, int, int, int]:
        """
        Calculate padding needed to make dimensions divisible by divisor.
        
        Args:
            height (int): Input height (frequency dimension)
            width (int): Input width (time dimension)
            divisor (int): The divisor for dimension alignment (default: 16 for 4 pooling layers)
            
        Returns:
            Tuple[int, int, int, int]: Padding values (left, right, top, bottom)
        """
        # Calculate target dimensions (nearest multiple of divisor)
        target_height = ((height + divisor - 1) // divisor) * divisor
        target_width = ((width + divisor - 1) // divisor) * divisor
        
        # Calculate total padding needed
        pad_height = target_height - height
        pad_width = target_width - width
        
        # Distribute padding evenly (put extra on right/bottom if odd)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        return (pad_left, pad_right, pad_top, pad_bottom)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic padding and cropping.
        
        This method implements the "Pad-Crop Wrapper" to handle dimension mismatches:
        1. Pad input to nearest multiple of 16 (for 4 pooling layers)
        2. Process through U-Net
        3. Crop output back to original dimensions
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 2, H, W]
                where H=103 (frequency), W=Time
        
        Returns:
            torch.Tensor: Output tensor of shape [B, 2, H, W] (same as input)
        """
        # Store original dimensions
        batch_size, channels, orig_height, orig_width = x.shape
        
        # Calculate padding
        divisor = 2 ** self.depth  # 2^4 = 16 for 4 pooling layers
        pad_left, pad_right, pad_top, pad_bottom = self._calculate_padding(
            orig_height, orig_width, divisor
        )
        
        # Pad input (using reflection padding to avoid edge artifacts)
        # Note: Reflection padding requires padding size < input dimension
        # Fall back to replication padding if reflection padding is not possible
        if (pad_left < orig_width and pad_right < orig_width and 
            pad_top < orig_height and pad_bottom < orig_height):
            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        else:
            # Use replication padding for very small inputs
            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        
        # Encoder path with skip connections
        skip_connections = []
        x_enc = x_padded
        
        for i in range(self.depth):
            x_enc = self.encoders[i](x_enc)
            skip_connections.append(x_enc)
            x_enc = self.pools[i](x_enc)
        
        # Bottleneck
        x_enc = self.bottleneck(x_enc)
        
        # Decoder path
        x_dec = x_enc
        for i in range(self.depth):
            x_dec = self.upconvs[i](x_dec)
            
            # Get corresponding skip connection
            skip = skip_connections[-(i + 1)]
            
            # Handle potential size mismatch due to odd dimensions
            if x_dec.shape != skip.shape:
                # Crop or pad to match skip connection size
                diff_h = skip.shape[2] - x_dec.shape[2]
                diff_w = skip.shape[3] - x_dec.shape[3]
                x_dec = F.pad(x_dec, [diff_w // 2, diff_w - diff_w // 2,
                                      diff_h // 2, diff_h - diff_h // 2])
            
            # Concatenate skip connection
            x_dec = torch.cat([skip, x_dec], dim=1)
            x_dec = self.decoders[i](x_dec)
        
        # Output layer (no activation - linear output for Real/Imag values)
        output_padded = self.out_conv(x_dec)
        
        # Crop back to original dimensions
        output = output_padded[:, :, pad_top:pad_top + orig_height, pad_left:pad_left + orig_width]
        
        return output


def test_model():
    """
    Test function to verify SpectrogramUNet functionality.
    
    Creates a random input tensor with problematic dimensions (103 frequency bins)
    and verifies that the output shape matches the input shape exactly.
    """
    print("=" * 70)
    print("Testing SpectrogramUNet")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = SpectrogramUNet(in_channels=2, out_channels=2, base_channels=64, depth=4)
    model.eval()  # Set to evaluation mode
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Test with problematic dimensions
    batch_size = 4
    freq_bins = 103  # Prime/odd number that breaks standard pooling
    time_steps = 156
    
    print(f"\nInput Shape: [B={batch_size}, C=2, F={freq_bins}, T={time_steps}]")
    
    # Create random input tensor
    x = torch.randn(batch_size, 2, freq_bins, time_steps)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output Shape: {list(output.shape)}")
    
    # Assertions
    assert output.shape == x.shape, \
        f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"
    
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == 2, "Channel count mismatch"
    assert output.shape[2] == freq_bins, "Frequency dimension mismatch"
    assert output.shape[3] == time_steps, "Time dimension mismatch"
    
    print("\n✓ All assertions passed!")
    
    # Additional tests with different time dimensions
    print("\n" + "-" * 70)
    print("Testing with various time dimensions:")
    print("-" * 70)
    
    test_time_dims = [64, 100, 128, 200, 256]
    
    for time_dim in test_time_dims:
        x_test = torch.randn(2, 2, freq_bins, time_dim)
        with torch.no_grad():
            out_test = model(x_test)
        
        assert out_test.shape == x_test.shape, \
            f"Failed for time_dim={time_dim}: {x_test.shape} != {out_test.shape}"
        
        print(f"  ✓ Time={time_dim:3d}: Input {list(x_test.shape)} -> Output {list(out_test.shape)}")
    
    print("\n" + "=" * 70)
    print("All tests passed successfully! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
