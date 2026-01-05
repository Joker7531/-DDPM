"""
Precompute STFT spectrograms for all dataset files.

This script computes STFT for all raw and clean signals and saves them
to disk to avoid real-time computation during training.

Usage:
    python precompute_stft.py --input Dataset --output Dataset_STFT
"""

import numpy as np
from pathlib import Path
from scipy import signal
from tqdm import tqdm
import argparse


def compute_and_save_stft(
    signal_path: Path,
    output_path: Path,
    fs: int = 500,
    nperseg: int = 256,
    noverlap: int = 224,
    nfft: int = 256
):
    """
    Compute STFT and save as [2, Freq, Time] tensor.
    
    Args:
        signal_path: Path to input .npy signal file
        output_path: Path to save STFT output
        fs: Sampling frequency
        nperseg: STFT window length
        noverlap: Overlap samples
        nfft: FFT length
    """
    # Load signal
    sig = np.load(signal_path)
    
    # Compute STFT
    # 显式指定所有参数以避免版本差异
    f, t, Zxx = signal.stft(
        sig,
        fs=fs,
        window='hann',      # 显式指定窗函数
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary='zeros',   # 默认值，显式指定
        padded=True,        # 默认值，显式指定
        return_onesided=True
    )
    
    # Split into real and imaginary
    real_part = np.real(Zxx).astype(np.float32)
    imag_part = np.imag(Zxx).astype(np.float32)
    
    # Stack as [2, Freq, Time]
    stft_tensor = np.stack([real_part, imag_part], axis=0)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, stft_tensor)
    
    return stft_tensor.shape


def precompute_dataset(
    input_root: str,
    output_root: str,
    fs: int = 500,
    nperseg: int = 256,
    hop_length: int = 32,
    nfft: int = 256
):
    """
    Precompute STFT for entire dataset.
    
    Args:
        input_root: Root directory of original dataset
        output_root: Root directory for STFT outputs
        fs: Sampling frequency
        nperseg: STFT window length
        hop_length: Hop length
        nfft: FFT length
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    noverlap = nperseg - hop_length
    
    print(f"STFT Parameters:")
    print(f"  fs = {fs} Hz")
    print(f"  n_fft = {nfft}")
    print(f"  nperseg = {nperseg}")
    print(f"  hop_length = {hop_length}")
    print(f"  noverlap = {noverlap}")
    print(f"  Frequency bins = {nfft // 2 + 1}")
    print()
    
    total_files = 0
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        for data_type in ['raw', 'clean']:
            input_dir = input_root / split / data_type
            output_dir = output_root / split / data_type
            
            if not input_dir.exists():
                print(f"  Skipping {input_dir} (not found)")
                continue
            
            # Find all .npy files
            files = sorted(list(input_dir.glob('*.npy')))
            
            if len(files) == 0:
                print(f"  No files found in {input_dir}")
                continue
            
            print(f"  Processing {len(files)} {data_type} files...")
            
            for file_path in tqdm(files, desc=f"  {split}/{data_type}"):
                output_path = output_dir / file_path.name
                
                try:
                    shape = compute_and_save_stft(
                        file_path,
                        output_path,
                        fs=fs,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        nfft=nfft
                    )
                    total_files += 1
                    
                except Exception as e:
                    print(f"\n  Error processing {file_path.name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"Precomputation Complete!")
    print(f"Total files processed: {total_files}")
    print(f"Output directory: {output_root}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Precompute STFT spectrograms')
    parser.add_argument('--input', type=str, default='Dataset',
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, default='Dataset_STFT',
                        help='Output directory for STFT files')
    parser.add_argument('--fs', type=int, default=500,
                        help='Sampling frequency (Hz)')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT length')
    parser.add_argument('--hop_length', type=int, default=64,
                        help='Hop length')
    
    args = parser.parse_args()
    
    precompute_dataset(
        input_root=args.input,
        output_root=args.output,
        fs=args.fs,
        nperseg=args.n_fft,
        hop_length=args.hop_length,
        nfft=args.n_fft
    )


if __name__ == "__main__":
    main()
