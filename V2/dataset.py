"""
Dataset Module for STFT-based EEG Signal Processing

This module provides PyTorch Dataset classes for loading and preprocessing
EEG signals in STFT domain for the SpectrogramUNet model.

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from scipy import signal


class EEGSTFTDataset(Dataset):
    """
    PyTorch Dataset for EEG signals with STFT transformation.
    
    This dataset loads raw and clean EEG signals from .npy files,
    applies STFT transformation on-the-fly, and returns dual-channel
    (Real/Imag) spectrograms.
    
    Args:
        root_dir (str): Root directory containing the dataset
        split (str): Dataset split ('train', 'val', or 'test')
        metadata_file (str): Path to metadata.csv file
        fs (int): Sampling frequency in Hz (default: 250)
        nperseg (int): Length of each segment for STFT (default: 256)
        noverlap (int): Number of points to overlap between segments (default: 128)
        nfft (Optional[int]): Length of FFT. If None, uses nperseg (default: None)
        transform (Optional[callable]): Optional transform to apply
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        metadata_file: str = 'metadata.csv',
        fs: int = 500,
        nperseg: int = 512,
        noverlap: int = 448,
        nfft: Optional[int] = None,
        transform: Optional[callable] = None
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft if nfft is not None else nperseg
        self.transform = transform
        
        # Find all raw files in the split directory
        raw_dir = self.root_dir / split / 'raw'
        clean_dir = self.root_dir / split / 'clean'
        
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
        
        # Get all raw files
        self.raw_files = sorted(list(raw_dir.glob('*_raw.npy')))
        
        if len(self.raw_files) == 0:
            raise ValueError(f"No raw files found in {raw_dir}")
        
        # Verify matching clean files exist
        self.clean_files = []
        for raw_file in self.raw_files:
            clean_file = clean_dir / raw_file.name.replace('_raw.npy', '_clean.npy')
            if not clean_file.exists():
                raise FileNotFoundError(f"Matching clean file not found: {clean_file}")
            self.clean_files.append(clean_file)
        
        print(f"Loaded {len(self.raw_files)} samples for split '{split}'")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.raw_files)
    
    def _compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform of the signal.
        
        Args:
            signal_data (np.ndarray): 1D time-domain signal
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Real and imaginary parts of STFT
                Both have shape [Freq, Time]
        """
        # Compute STFT
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=True
        )
        
        # Split into real and imaginary parts
        real_part = np.real(Zxx).astype(np.float32)
        imag_part = np.imag(Zxx).astype(np.float32)
        
        return real_part, imag_part
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - raw_stft: Raw signal STFT [2, Freq, Time]
                - clean_stft: Clean signal STFT [2, Freq, Time]
        """
        # Get file paths
        raw_path = self.raw_files[idx]
        clean_path = self.clean_files[idx]
        
        # Load signals
        raw_signal = np.load(raw_path)
        clean_signal = np.load(clean_path)
        
        # Compute STFT for both signals
        raw_real, raw_imag = self._compute_stft(raw_signal)
        clean_real, clean_imag = self._compute_stft(clean_signal)
        
        # Stack real and imaginary parts as channels [2, Freq, Time]
        raw_stft = np.stack([raw_real, raw_imag], axis=0)
        clean_stft = np.stack([clean_real, clean_imag], axis=0)
        
        # Convert to torch tensors
        raw_stft = torch.from_numpy(raw_stft)
        clean_stft = torch.from_numpy(clean_stft)
        
        # Apply optional transforms
        if self.transform:
            raw_stft = self.transform(raw_stft)
            clean_stft = self.transform(clean_stft)
        
        return raw_stft, clean_stft


class PrecomputedSTFTDataset(Dataset):
    """
    Dataset for loading precomputed STFT spectrograms from disk.
    
    Uses sliding window to extract fixed-length segments from each file,
    ensuring all data is utilized.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        segment_length: int = 625,  # ~20s @ hop_length=32, fs=500
        stride: Optional[int] = None  # If None, uses segment_length (no overlap)
    ):
        """
        Args:
            root_dir (str): Path to the directory containing train/val/test splits
            split (str): Which split to use ('train', 'val', or 'test')
            transform (optional): Optional transform to apply to the spectrograms
            segment_length (int): Number of time frames per segment
            stride (int): Stride for sliding window. If None, defaults to segment_length
        """
        super().__init__()
        
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.segment_length = segment_length
        self.stride = stride if stride is not None else segment_length
        
        # Find all STFT pairs
        self.raw_files = sorted((self.root_dir / 'raw').glob('*.npy'))
        self.clean_files = sorted((self.root_dir / 'clean').glob('*.npy'))
        
        assert len(self.raw_files) == len(self.clean_files), \
            "Mismatch between number of raw and clean files"
        
        # Pre-compute all window indices: (file_idx, start_frame)
        self.window_indices = []
        total_frames = 0
        
        for file_idx in range(len(self.raw_files)):
            # Load file to get length
            stft_data = np.load(self.raw_files[file_idx])
            _, _, time_len = stft_data.shape
            total_frames += time_len
            
            # Create sliding windows
            if time_len < segment_length:
                # If file is shorter than segment, pad it and use as one segment
                self.window_indices.append((file_idx, 0))
            else:
                # Sliding window with stride
                for start in range(0, time_len - segment_length + 1, self.stride):
                    self.window_indices.append((file_idx, start))
        
        print(f"Loaded {len(self.raw_files)} precomputed STFT files for split '{split}'")
        print(f"  Segment length: {segment_length} frames (~{segment_length*32/500:.1f}s @ hop=32, fs=500)")
        print(f"  Stride: {self.stride} frames (~{self.stride*32/500:.1f}s)")
        print(f"  Total windows: {len(self.window_indices)} (from {total_frames} total frames)")
    
    def __len__(self) -> int:
        """Return the total number of windowed segments."""
        return len(self.window_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a windowed segment from the dataset.
        
        Args:
            idx (int): Index of the window
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - raw_stft: Raw signal STFT segment [2, Freq, segment_length]
                - clean_stft: Clean signal STFT segment [2, Freq, segment_length]
        """
        # Get file index and start frame for this window
        file_idx, start_frame = self.window_indices[idx]
        
        # Load precomputed STFT spectrograms
        raw_stft = np.load(self.raw_files[file_idx])
        clean_stft = np.load(self.clean_files[file_idx])
        
        # Extract segment
        _, _, time_len = raw_stft.shape
        
        if time_len < self.segment_length:
            # Pad if file is shorter than segment length
            pad_width = ((0, 0), (0, 0), (0, self.segment_length - time_len))
            raw_stft = np.pad(raw_stft, pad_width, mode='constant')
            clean_stft = np.pad(clean_stft, pad_width, mode='constant')
        else:
            # Extract window
            end_frame = start_frame + self.segment_length
            raw_stft = raw_stft[:, :, start_frame:end_frame]
            clean_stft = clean_stft[:, :, start_frame:end_frame]
        
        # Convert to torch tensors
        raw_stft = torch.from_numpy(raw_stft).float()
        clean_stft = torch.from_numpy(clean_stft).float()
        
        # Apply optional transforms
        if self.transform:
            raw_stft = self.transform(raw_stft)
            clean_stft = self.transform(clean_stft)
        
        return raw_stft, clean_stft


def test_dataset():
    """Test function for EEG STFT Dataset."""
    print("=" * 70)
    print("Testing EEGSTFTDataset")
    print("=" * 70)
    
    # This is a dummy test - adapt paths to your actual data
    dataset_path = "../../dataset"
    
    try:
        # Try to load the dataset
        dataset = EEGSTFTDataset(
            root_dir=dataset_path,
            split='train',
            fs=250,
            nperseg=256,
            noverlap=128
        )
        
        print(f"\nDataset Size: {len(dataset)}")
        
        # Get a sample
        raw_stft, clean_stft = dataset[0]
        
        print(f"\nSample Shapes:")
        print(f"  Raw STFT:   {list(raw_stft.shape)}")
        print(f"  Clean STFT: {list(clean_stft.shape)}")
        
        assert raw_stft.shape[0] == 2, "Expected 2 channels (Real/Imag)"
        assert clean_stft.shape[0] == 2, "Expected 2 channels (Real/Imag)"
        assert raw_stft.shape == clean_stft.shape, "Shape mismatch"
        
        print("\n✓ Dataset test passed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Dataset not found: {e}")
        print("This is expected if dataset is not yet prepared.")
        print("Use the prepare_dataset.py script to create the dataset first.")


if __name__ == "__main__":
    test_dataset()
