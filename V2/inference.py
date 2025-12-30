"""
Inference and Evaluation Script for SpectrogramUNet

This module provides functions for:
1. Loading trained models
2. Running inference on test data
3. Converting STFT back to time-domain signals
4. Evaluating reconstruction quality
5. Visualizing results

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SpectrogramUNet
from dataset import EEGSTFTDataset


def stft_to_signal(
    stft_real: np.ndarray,
    stft_imag: np.ndarray,
    fs: int = 250,
    nperseg: int = 256,
    noverlap: int = 128,
    nfft: Optional[int] = None
) -> np.ndarray:
    """
    Convert STFT spectrogram back to time-domain signal using inverse STFT.
    
    Args:
        stft_real (np.ndarray): Real part of STFT [Freq, Time]
        stft_imag (np.ndarray): Imaginary part of STFT [Freq, Time]
        fs (int): Sampling frequency (default: 250)
        nperseg (int): Length of each segment (default: 256)
        noverlap (int): Number of overlapping points (default: 128)
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


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        clean (np.ndarray): Clean signal
        noisy (np.ndarray): Noisy signal
        
    Returns:
        float: SNR in dB
    """
    noise = clean - noisy
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute various reconstruction metrics.
    
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
    snr = compute_snr(target, pred)
    
    # Correlation
    correlation = np.corrcoef(pred, target)[0, 1]
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'snr': float(snr),
        'correlation': float(correlation)
    }


class Inferencer:
    """
    Inference class for SpectrogramUNet.
    
    Args:
        model_path (str): Path to trained model checkpoint
        device (str): Device to run inference on ('cuda' or 'cpu')
        config (Optional[dict]): Model configuration (loaded from checkpoint if None)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        config: Optional[dict] = None
    ):
        self.device = device
        
        # Load checkpoint (PyTorch 2.6+ requires weights_only=False for full checkpoints)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get config from checkpoint or use provided
        if config is None:
            config = checkpoint.get('config', {})
        
        # Create model
        self.model = SpectrogramUNet(
            in_channels=config.get('in_channels', 2),
            out_channels=config.get('out_channels', 2),
            base_channels=config.get('base_channels', 64),
            depth=config.get('depth', 4)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Store STFT parameters
        self.fs = config.get('fs', 500)
        self.nperseg = config.get('n_fft', 512)
        self.noverlap = config.get('n_fft', 512) - config.get('hop_length', 64)
        self.nfft = config.get('n_fft', 512)
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    @torch.no_grad()
    def denoise_stft(self, raw_stft: torch.Tensor) -> torch.Tensor:
        """
        Denoise STFT spectrogram.
        
        Args:
            raw_stft (torch.Tensor): Raw STFT [B, 2, F, T] or [2, F, T]
            
        Returns:
            torch.Tensor: Denoised STFT (same shape as input)
        """
        # Add batch dimension if needed
        if raw_stft.ndim == 3:
            raw_stft = raw_stft.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Move to device
        raw_stft = raw_stft.to(self.device)
        
        # Inference
        pred_stft = self.model(raw_stft)
        
        # Remove batch dimension if added
        if squeeze_output:
            pred_stft = pred_stft.squeeze(0)
        
        return pred_stft.cpu()
    
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
            np.ndarray: Reconstructed time-domain signal
        """
        _, _, time_len = raw_stft.shape
        
        # Calculate output signal length (approximate, will be determined by ISTFT)
        # Each STFT frame corresponds to hop_length samples
        hop_length = self.nperseg - self.noverlap
        approx_signal_len = (time_len - 1) * hop_length + self.nperseg
        
        # Initialize output signal and weight accumulator for overlap-add
        output_signal = np.zeros(approx_signal_len, dtype=np.float32)
        weight_sum = np.zeros(approx_signal_len, dtype=np.float32)
        
        # Create Hann window for smooth blending
        window = np.hanning(segment_length * hop_length)
        
        # Generate window indices
        if time_len < segment_length:
            # If shorter than segment, process as single segment with padding
            window_starts = [0]
        else:
            window_starts = list(range(0, time_len - segment_length + 1, stride))
        
        print(f"\n  Total STFT frames: {time_len}")
        print(f"  Processing {len(window_starts)} overlapping windows...")
        print(f"  Window size: {segment_length} frames (~{segment_length*hop_length/self.fs:.1f}s)")
        print(f"  Stride: {stride} frames (~{stride*hop_length/self.fs:.1f}s)")
        
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
            denoised_batch = self.denoise_stft(batch_tensor).numpy()
            
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
    
    def denoise_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Denoise time-domain signal.
        
        Performs: Time-domain -> STFT -> U-Net -> ISTFT -> Time-domain
        
        Args:
            raw_signal (np.ndarray): Raw time-domain signal (1D)
            
        Returns:
            np.ndarray: Denoised time-domain signal
        """
        # Compute STFT
        f, t, Zxx = signal.stft(
            raw_signal,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft if self.nfft is not None else self.nperseg,
            return_onesided=True
        )
        
        # Split into real and imaginary parts
        real_part = np.real(Zxx).astype(np.float32)
        imag_part = np.imag(Zxx).astype(np.float32)
        
        # Stack as tensor [2, F, T]
        raw_stft = torch.from_numpy(np.stack([real_part, imag_part], axis=0))
        
        # Denoise
        pred_stft = self.denoise_stft(raw_stft).numpy()
        
        # Convert back to time-domain
        denoised_signal = stft_to_signal(
            pred_stft[0], pred_stft[1],
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )
        
        return denoised_signal
    
    def evaluate_dataset(self, dataset: EEGSTFTDataset, save_dir: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.
        
        Args:
            dataset (EEGSTFTDataset): Dataset to evaluate
            save_dir (Optional[str]): Directory to save results (default: None)
            
        Returns:
            Dict[str, float]: Average metrics across dataset
        """
        all_metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'snr': [],
            'correlation': []
        }
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        for idx in tqdm(range(len(dataset))):
            # Get data
            raw_stft, clean_stft = dataset[idx]
            
            # Denoise
            pred_stft = self.denoise_stft(raw_stft)
            
            # Convert to time-domain
            raw_signal = stft_to_signal(
                raw_stft[0].numpy(), raw_stft[1].numpy(),
                self.fs, self.nperseg, self.noverlap, self.nfft
            )
            
            clean_signal = stft_to_signal(
                clean_stft[0].numpy(), clean_stft[1].numpy(),
                self.fs, self.nperseg, self.noverlap, self.nfft
            )
            
            pred_signal = stft_to_signal(
                pred_stft[0].numpy(), pred_stft[1].numpy(),
                self.fs, self.nperseg, self.noverlap, self.nfft
            )
            
            # Compute metrics
            metrics = compute_metrics(pred_signal, clean_signal)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Save if requested
            if save_dir and idx < 10:  # Save first 10 samples
                self._save_visualization(
                    raw_signal, clean_signal, pred_signal,
                    save_dir / f'sample_{idx:03d}.png'
                )
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
        
        # Combine
        result_metrics = {**avg_metrics, **std_metrics}
        
        # Print results
        print("\n" + "=" * 70)
        print("Evaluation Results:")
        print("=" * 70)
        for key, value in avg_metrics.items():
            std = std_metrics[f"{key}_std"]
            print(f"  {key.upper():12s}: {value:.4f} ± {std:.4f}")
        print("=" * 70)
        
        return result_metrics
    
    def _save_visualization(
        self,
        raw: np.ndarray,
        clean: np.ndarray,
        pred: np.ndarray,
        save_path: Path
    ):
        """Save visualization of signals."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        time_axis = np.arange(len(raw)) / self.fs
        
        axes[0].plot(time_axis, raw, 'b-', alpha=0.7, linewidth=0.5)
        axes[0].set_title('Raw Signal (Noisy)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_axis, clean, 'g-', alpha=0.7, linewidth=0.5)
        axes[1].set_title('Clean Signal (Ground Truth)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time_axis, pred, 'r-', alpha=0.7, linewidth=0.5)
        axes[2].set_title('Predicted Signal (Denoised)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with SpectrogramUNet')
    parser.add_argument('--model', type=str, default='./checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input (.npy STFT file or dataset root directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for single file inference (.npy)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (only for directory input)')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results (only for directory input)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = Inferencer(
        model_path=args.model,
        device=args.device if torch.cuda.is_available() else 'cpu'
    )
    
    data_path = Path(args.data)
    
    # Check if input is a single .npy file or a directory
    if data_path.is_file() and data_path.suffix == '.npy':
        # Single file inference with sliding window + overlap-add
        print(f"\n{'='*70}")
        print(f"Processing single file: {data_path.name}")
        print(f"{'='*70}")
        
        # Load STFT
        raw_stft = np.load(data_path)
        print(f"  Input STFT shape: {raw_stft.shape}")
        
        # Denoise using sliding window with overlap-add
        print("\n  Running sliding window inference with overlap-add...")
        denoised_signal = inferencer.denoise_full_stft(
            raw_stft,
            segment_length=625,  # ~20s windows
            stride=312,          # ~10s stride (50% overlap)
            batch_size=8
        )
        
        # Save denoised time-domain signal
        output_path = args.output if args.output else data_path.parent / f"{data_path.stem}_denoised_signal.npy"
        np.save(output_path, denoised_signal)
        
        print(f"\n{'='*70}")
        print(f"✓ Denoised signal saved to {output_path.name}")
        print(f"  Output shape: {denoised_signal.shape}")
        print(f"  Duration: {len(denoised_signal)/inferencer.fs:.2f}s @ {inferencer.fs}Hz")
        print(f"{'='*70}")
        
    elif data_path.is_dir():
        # Directory inference - evaluate on dataset
        print(f"\nEvaluating on dataset: {data_path}")
        
        from dataset import PrecomputedSTFTDataset
        
        # Load test dataset
        test_dataset = PrecomputedSTFTDataset(
            root_dir=args.data,
            split=args.split,
            segment_length=625  # Use default segment length
        )
        
        # Evaluate
        metrics = inferencer.evaluate_dataset(test_dataset, save_dir=args.save_dir)
        
        # Save metrics
        import json
        metrics_path = Path(args.save_dir) / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✓ Results saved to {args.save_dir}")
    
    else:
        raise ValueError(f"Invalid input: {data_path}. Must be a .npy file or directory.")


if __name__ == "__main__":
    main()
