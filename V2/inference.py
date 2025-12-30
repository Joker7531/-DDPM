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
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
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
    parser.add_argument('--data', type=str, default='../../dataset',
                        help='Path to dataset root')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = Inferencer(
        model_path=args.model,
        device=args.device if torch.cuda.is_available() else 'cpu'
    )
    
    # Load test dataset
    test_dataset = EEGSTFTDataset(
        root_dir=args.data,
        split=args.split,
        fs=inferencer.fs,
        nperseg=inferencer.nperseg,
        noverlap=inferencer.noverlap
    )
    
    # Evaluate
    metrics = inferencer.evaluate_dataset(test_dataset, save_dir=args.save_dir)
    
    # Save metrics
    import json
    metrics_path = Path(args.save_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n✓ Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
