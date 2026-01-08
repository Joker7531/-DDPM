"""
UAR-ACSSNet: 单通道 EEG 去伪影/重建系统

Main components:
    - datasets: EEGPairDataset, build_dataloaders
    - signal: STFTProcessor
    - models: UAR_ACSSNet
    - train: train, validate, compute_losses
    - configs: get_default_config
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .datasets import EEGPairDataset, build_dataloaders
from .signal_processing import STFTProcessor
from .models import UAR_ACSSNet
from .train import train, validate, compute_losses, set_seed
from .configs import get_default_config

__all__ = [
    "EEGPairDataset",
    "build_dataloaders",
    "STFTProcessor",
    "UAR_ACSSNet",
    "train",
    "validate",
    "compute_losses",
    "set_seed",
    "get_default_config",
]
