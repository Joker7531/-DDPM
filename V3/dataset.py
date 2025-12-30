"""
Dataset Classes for Residual Noise Prediction (V3)

Returns noise residual (Raw - Clean) as target instead of clean signal.

Author: Expert PyTorch Engineer
Date: 2025-12-30
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any


class ResidualSTFTDataset(Dataset):
    """
    Dataset for loading precomputed STFT spectrograms and returning noise residual.
    
    Uses sliding window to extract fixed-length segments.
    Returns (raw_stft, noise_stft) where noise_stft = raw_stft - clean_stft.
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
        print(f"  Target: Noise Residual (Raw - Clean)")
    
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
                - noise_stft: Noise residual (raw - clean) [2, Freq, segment_length]
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
        
        # Compute noise residual: noise = raw - clean
        noise_stft = raw_stft - clean_stft
        
        # Convert to torch tensors
        raw_stft = torch.from_numpy(raw_stft).float()
        noise_stft = torch.from_numpy(noise_stft).float()
        
        # Apply optional transforms
        if self.transform:
            raw_stft = self.transform(raw_stft)
            noise_stft = self.transform(noise_stft)
        
        return raw_stft, noise_stft
    
    def get_clean_stft(self, idx: int) -> torch.Tensor:
        """
        Get clean STFT for a given index (for validation/evaluation).
        
        Args:
            idx (int): Index of the window
            
        Returns:
            torch.Tensor: Clean STFT segment [2, Freq, segment_length]
        """
        file_idx, start_frame = self.window_indices[idx]
        clean_stft = np.load(self.clean_files[file_idx])
        
        _, _, time_len = clean_stft.shape
        
        if time_len < self.segment_length:
            pad_width = ((0, 0), (0, 0), (0, self.segment_length - time_len))
            clean_stft = np.pad(clean_stft, pad_width, mode='constant')
        else:
            end_frame = start_frame + self.segment_length
            clean_stft = clean_stft[:, :, start_frame:end_frame]
        
        return torch.from_numpy(clean_stft).float()


def test_dataset():
    """Test function for ResidualSTFTDataset."""
    print("=" * 70)
    print("Testing ResidualSTFTDataset (V3)")
    print("=" * 70)
    
    # Dummy test paths
    dataset_path = "../Dataset_STFT"
    
    try:
        dataset = ResidualSTFTDataset(
            root_dir=dataset_path,
            split='train',
            segment_length=625,
            stride=312
        )
        
        print(f"\nDataset Size: {len(dataset)} windows")
        
        # Get a sample
        raw_stft, noise_stft = dataset[0]
        
        print(f"\nSample Shapes:")
        print(f"  Raw STFT:   {list(raw_stft.shape)}")
        print(f"  Noise STFT: {list(noise_stft.shape)}")
        
        # Verify noise computation
        clean_stft = dataset.get_clean_stft(0)
        reconstructed_clean = raw_stft - noise_stft
        
        assert torch.allclose(reconstructed_clean, clean_stft, atol=1e-6), \
            "Noise residual verification failed!"
        print("\n✓ Verified: clean = raw - noise")
        
        print("\n✓ Dataset test passed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Dataset not found: {e}")
        print("This is expected if dataset is not yet prepared.")


if __name__ == "__main__":
    test_dataset()
