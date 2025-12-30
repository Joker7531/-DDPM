"""
SpectrogramUNet Package for EEG Signal Denoising

A complete implementation of 2D U-Net for STFT-domain EEG signal reconstruction.

Author: Expert PyTorch Engineer
Date: 2025-12-30
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Expert PyTorch Engineer"

from .model import SpectrogramUNet, DoubleConv
from .dataset import EEGSTFTDataset, PrecomputedSTFTDataset
from .inference import Inferencer, stft_to_signal, compute_metrics, compute_snr

__all__ = [
    'SpectrogramUNet',
    'DoubleConv',
    'EEGSTFTDataset',
    'PrecomputedSTFTDataset',
    'Inferencer',
    'stft_to_signal',
    'compute_metrics',
    'compute_snr',
]
