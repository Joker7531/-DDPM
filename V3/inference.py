"""
Inference Script for Residual Noise Prediction U-Net (V3)

Key inference formula: Clean_Pred = Raw - Model(Raw)

Features:
- Sliding window with overlap-add for full signal reconstruction
- Hann window weighting for smooth transitions
- Batch processing for efficiency

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ResidualNoiseUNet


def stft_to_signal(
    stft_real: np.ndarray,
    stft_imag: np.ndarray,
    fs: int = 500,
    nperseg: int = 512,
    noverlap: int = 480,
    nfft: Optional[int] = None
) -> np.ndarray:
    """
    Convert STFT spectrogram back to time-domain signal using inverse STFT.
    
    Args:
        stft_real (np.ndarray): Real part of STFT [Freq, Time]
        stft_imag (np.ndarray): Imaginary part of STFT [Freq, Time]
        fs (int): Sampling frequency (default: 500)
        nperseg (int): Length of each segment (default: 512)
        noverlap (int): Number of overlapping points (default: 480)
        nfft (Optional[int]): FFT length (default: None, uses nperseg)
        
    Returns:
        np.ndarray: Reconstructed time-domain signal
    """
    # Combine real and imaginary parts
    Zxx = stft_real + 1j * stft_imag
    
    # Inverse STFT
    _, reconstructed = signal.istft(
        Zxx,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft if nfft is not None else nperseg
    )
    
    return reconstructed


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute reconstruction metrics.
    
    Args:
        pred (np.ndarray): Predicted signal
        target (np.ndarray): Target (clean) signal
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Ensure same length
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(pred - target))
    
    # SNR
    noise = pred - target
    signal_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Correlation
    correlation = np.corrcoef(pred, target)[0, 1]
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'snr': float(snr),
        'correlation': float(correlation)
    }


class ResidualInferencer:
    """
    Inference class for ResidualNoiseUNet.
    
    Uses residual formula: Clean = Raw - Model(Raw)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        config: Optional[dict] = None
    ):
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get config from checkpoint or use provided
        if config is None:
            config = checkpoint.get('config', {})
        
        # Create model
        self.model = ResidualNoiseUNet(
            in_channels=config.get('in_channels', 2),
            out_channels=config.get('out_channels', 2),
            base_channels=config.get('base_channels', 32),
            depth=config.get('depth', 4)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Store STFT parameters
        self.fs = config.get('fs', 500)
        self.nperseg = config.get('n_fft', 512)
        self.noverlap = config.get('n_fft', 512) - config.get('hop_length', 32)
        self.nfft = config.get('n_fft', 512)
        
        print(f"✓ Loaded ResidualNoiseUNet from {model_path}")
        print(f"  Model type: {config.get('model_type', 'ResidualNoiseUNet')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    @torch.no_grad()
    def predict_noise(self, raw_stft: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from raw STFT.
        
        Args:
            raw_stft (torch.Tensor): Raw STFT [B, 2, F, T] or [2, F, T]
            
        Returns:
            torch.Tensor: Predicted noise (same shape as input)
        """
        # Add batch dimension if needed
        if raw_stft.ndim == 3:
            raw_stft = raw_stft.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Move to device
        raw_stft = raw_stft.to(self.device)
        
        # Predict noise
        pred_noise = self.model(raw_stft)
        
        # Remove batch dimension if added
        if squeeze_output:
            pred_noise = pred_noise.squeeze(0)
        
        return pred_noise.cpu()
    
    @torch.no_grad()
    def denoise(self, raw_stft: torch.Tensor) -> torch.Tensor:
        """
        Denoise STFT using residual formula: Clean = Raw - Noise
        
        Args:
            raw_stft (torch.Tensor): Raw STFT [B, 2, F, T] or [2, F, T]
            
        Returns:
            torch.Tensor: Denoised STFT (same shape as input)
        """
        # Add batch dimension if needed
        squeeze_output = False
        if raw_stft.ndim == 3:
            raw_stft = raw_stft.unsqueeze(0)
            squeeze_output = True
        
        raw_stft_device = raw_stft.to(self.device)
        
        # Predict noise
        pred_noise = self.model(raw_stft_device)
        
        # Residual subtraction: Clean = Raw - Noise
        clean_pred = raw_stft_device - pred_noise
        
        if squeeze_output:
            clean_pred = clean_pred.squeeze(0)
        
        return clean_pred.cpu()
    
    @torch.no_grad()
    def denoise_full_stft(
        self,
        raw_stft: np.ndarray,
        segment_length: int = 625,
        stride: int = 312,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Denoise full STFT using sliding window with overlap-add reconstruction.
        
        Args:
            raw_stft (np.ndarray): Full raw STFT [2, Freq, Time]
            segment_length (int): Length of each window
            stride (int): Stride for sliding window
            batch_size (int): Batch size for inference
            
        Returns:
            np.ndarray: Reconstructed denoised time-domain signal
        """
        _, _, time_len = raw_stft.shape
        
        # Calculate output signal length
        hop_length = self.nperseg - self.noverlap
        approx_signal_len = (time_len - 1) * hop_length + self.nperseg
        
        # Initialize output signal and weight accumulator
        output_signal = np.zeros(approx_signal_len, dtype=np.float32)
        weight_sum = np.zeros(approx_signal_len, dtype=np.float32)
        
        # Create Hann window for smooth blending
        window = np.hanning(segment_length * hop_length)
        
        # Generate window indices
        if time_len < segment_length:
            window_starts = [0]
        else:
            window_starts = list(range(0, time_len - segment_length + 1, stride))
        
        print(f"\n  Total STFT frames: {time_len}")
        print(f"  Processing {len(window_starts)} overlapping windows...")
        print(f"  Window size: {segment_length} frames (~{segment_length*hop_length/self.fs:.1f}s)")
        print(f"  Stride: {stride} frames (~{stride*hop_length/self.fs:.1f}s)")
        print(f"  Inference: Clean = Raw - Model(Raw)")
        
        # Process windows in batches
        for batch_idx in tqdm(range(0, len(window_starts), batch_size), desc="  Denoising"):
            batch_starts = window_starts[batch_idx:batch_idx + batch_size]
            batch_segments = []
            
            # Extract segments for batch
            for start in batch_starts:
                end = start + segment_length
                if end <= time_len:
                    segment = raw_stft[:, :, start:end]
                else:
                    # Pad if needed
                    segment = raw_stft[:, :, start:]
                    pad_width = ((0, 0), (0, 0), (0, segment_length - segment.shape[2]))
                    segment = np.pad(segment, pad_width, mode='constant')
                
                batch_segments.append(segment)
            
            # Convert to tensor and denoise batch
            batch_tensor = torch.from_numpy(np.stack(batch_segments, axis=0)).float()
            denoised_batch = self.denoise(batch_tensor).numpy()
            
            # Convert each denoised segment to time-domain and overlap-add
            for i, start in enumerate(batch_starts):
                # ISTFT to time-domain
                segment_signal = stft_to_signal(
                    denoised_batch[i, 0],
                    denoised_batch[i, 1],
                    fs=self.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.nfft
                )
                
                # Calculate position in output signal
                signal_start = start * hop_length
                signal_end = signal_start + len(segment_signal)
                
                # Ensure we don't exceed output buffer
                if signal_end > len(output_signal):
                    signal_end = len(output_signal)
                    segment_signal = segment_signal[:signal_end - signal_start]
                
                # Create appropriate window for this segment
                seg_window = window[:len(segment_signal)]
                
                # Overlap-add with windowing
                output_signal[signal_start:signal_end] += segment_signal * seg_window
                weight_sum[signal_start:signal_end] += seg_window
        
        # Normalize by accumulated weights
        valid_mask = weight_sum > 1e-8
        output_signal[valid_mask] /= weight_sum[valid_mask]
        
        print(f"  Reconstructed signal length: {len(output_signal)} samples ({len(output_signal)/self.fs:.2f}s)")
        
        return output_signal
    
    def visualize_denoising(
        self,
        raw_stft: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize denoising on a sample.
        """
        # Predict noise and clean
        raw_tensor = torch.from_numpy(raw_stft).float()
        pred_noise = self.predict_noise(raw_tensor).numpy()
        clean_pred = raw_stft - pred_noise
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Raw STFT
        axes[0, 0].imshow(raw_stft[0], aspect='auto', origin='lower', cmap='RdBu')
        axes[0, 0].set_title('Raw STFT (Real)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].imshow(raw_stft[1], aspect='auto', origin='lower', cmap='RdBu')
        axes[0, 1].set_title('Raw STFT (Imag)', fontweight='bold')
        
        # Predicted Noise
        axes[1, 0].imshow(pred_noise[0], aspect='auto', origin='lower', cmap='RdBu')
        axes[1, 0].set_title('Predicted Noise (Real)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].imshow(pred_noise[1], aspect='auto', origin='lower', cmap='RdBu')
        axes[1, 1].set_title('Predicted Noise (Imag)', fontweight='bold')
        
        # Denoised STFT
        axes[2, 0].imshow(clean_pred[0], aspect='auto', origin='lower', cmap='RdBu')
        axes[2, 0].set_title('Denoised STFT (Real)', fontweight='bold')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Frequency')
        
        axes[2, 1].imshow(clean_pred[1], aspect='auto', origin='lower', cmap='RdBu')
        axes[2, 1].set_title('Denoised STFT (Imag)', fontweight='bold')
        axes[2, 1].set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with ResidualNoiseUNet (V3)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input (.npy STFT file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for denoised signal (.npy)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = ResidualInferencer(
        model_path=args.model,
        device=args.device if torch.cuda.is_available() else 'cpu'
    )
    
    data_path = Path(args.data)
    
    if data_path.is_file() and data_path.suffix == '.npy':
        print(f"\n{'='*70}")
        print(f"Processing: {data_path.name}")
        print(f"{'='*70}")
        
        # Load STFT
        raw_stft = np.load(data_path)
        print(f"  Input STFT shape: {raw_stft.shape}")
        
        # Visualize if requested
        if args.visualize:
            vis_path = data_path.parent / f"{data_path.stem}_visualization.png"
            # Use first segment for visualization
            segment = raw_stft[:, :, :625] if raw_stft.shape[2] > 625 else raw_stft
            inferencer.visualize_denoising(segment, save_path=str(vis_path))
        
        # Denoise using sliding window with overlap-add
        print("\n  Running residual denoising (Clean = Raw - Noise)...")
        denoised_signal = inferencer.denoise_full_stft(
            raw_stft,
            segment_length=625,  # ~20s windows
            stride=312,          # ~10s stride (50% overlap)
            batch_size=8
        )
        
        # Save denoised time-domain signal
        output_path = args.output if args.output else data_path.parent / f"{data_path.stem}_denoised.npy"
        np.save(output_path, denoised_signal)
        
        print(f"\n{'='*70}")
        print(f"✓ Denoised signal saved to {output_path}")
        print(f"  Output shape: {denoised_signal.shape}")
        print(f"  Duration: {len(denoised_signal)/inferencer.fs:.2f}s @ {inferencer.fs}Hz")
        print(f"{'='*70}")
    
    else:
        raise ValueError(f"Invalid input: {data_path}. Must be a .npy file.")


if __name__ == "__main__":
    main()
