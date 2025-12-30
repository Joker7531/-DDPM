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
        
        # Load metadata
        metadata_path = self.root_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by split
        self.data_info = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        if len(self.data_info) == 0:
            raise ValueError(f"No data found for split '{split}'")
        
        print(f"Loaded {len(self.data_info)} samples for split '{split}'")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_info)
    
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
        # Get file information
        row = self.data_info.iloc[idx]
        filename = row['filename']
        
        # Construct file paths
        raw_path = self.root_dir / self.split / 'raw' / f"{filename}_raw.npy"
        clean_path = self.root_dir / self.split / 'clean' / f"{filename}_clean.npy"
        
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
    Dataset for pre-computed STFT spectrograms.
    
    Use this if you want to precompute and save STFT spectrograms
    to disk for faster training.
    
    Args:
        root_dir (str): Root directory containing precomputed STFT files
        split (str): Dataset split ('train', 'val', or 'test')
        transform (Optional[callable]): Optional transform to apply
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # Find all STFT pairs
        self.raw_files = sorted((self.root_dir / 'raw').glob('*.npy'))
        self.clean_files = sorted((self.root_dir / 'clean').glob('*.npy'))
        
        assert len(self.raw_files) == len(self.clean_files), \
            "Mismatch between number of raw and clean files"
        
        print(f"Loaded {len(self.raw_files)} precomputed STFT pairs for split '{split}'")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.raw_files)
    
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
        # Load precomputed STFT spectrograms
        raw_stft = np.load(self.raw_files[idx])
        clean_stft = np.load(self.clean_files[idx])
        
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
