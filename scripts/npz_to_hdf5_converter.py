#!/usr/bin/env python3
"""
NPZ to HDF5 Converter for TERS Dataset

Converts a directory of .npz files to a single HDF5 file with pre-computed:
- uniform_channels (100 and 400 channel versions)
- target_image (molecule circular mask)

Usage:
    python npz_to_hdf5_converter.py --input_dir /path/to/npz --output /path/to/output.h5
    
    # With all options:
    python npz_to_hdf5_converter.py \
        --input_dir /path/to/train \
        --output /path/to/train.h5 \
        --circle_radius 5 \
        --sg_ch \
        --compression gzip \
        --compression_level 4

Author: Auto-generated
Date: 2025-12-05
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.xyz_to_label import molecule_circular_image


# Atomic symbols mapping
ATOMIC_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
    6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
}


def uniform_channels(spectrums, frequencies, num_channels=400):
    """
    Bin spectrums into uniform frequency channels.
    
    Args:
        spectrums: (H, W, N_freq) array of spectral images
        frequencies: (N_freq,) array of frequencies
        num_channels: Number of output channels (100 or 400)
    
    Returns:
        (H, W, num_channels) array of binned channels
    """
    max_freq = 4000
    step = max_freq // num_channels
    grid_size = spectrums.shape[0]
    channels = np.zeros((grid_size, grid_size, num_channels), dtype=np.float32)

    count = 0
    for i in range(1, max_freq, step):
        indices = (frequencies > i) & (frequencies < i + step)
        selected_spectrums = spectrums[:, :, indices]
        if np.all(selected_spectrums == 0) or selected_spectrums.size == 0:
            count += 1
            continue
        channels[:, :, count] = np.mean(selected_spectrums, axis=2)
        count += 1

    return channels


def compute_target_image(atom_pos, atomic_numbers, sg_ch=True, circle_radius=5):
    """
    Compute the target circular mask image from atom positions.
    
    Args:
        atom_pos: (N_atoms, 3) array of atom positions
        atomic_numbers: (N_atoms,) array of atomic numbers
        sg_ch: Single channel (True) or multi-channel (False)
        circle_radius: Radius of circles in pixels
    
    Returns:
        (C, H, W) target image array
    """
    # Build xyz string for molecule_circular_image
    t = list(zip(atomic_numbers, atom_pos))
    text = f"{len(t)}\nComment\n"
    for atom, pos in t:
        pos_str = "\t".join(f"{coord:.6f}" for coord in pos)
        text += ATOMIC_SYMBOLS.get(int(atom), "X") + "\t" + pos_str + "\n"
    
    target_image = molecule_circular_image(text, flag=sg_ch, circle_radius=circle_radius)
    return target_image


def convert_npz_to_hdf5(input_dir, output_path, frequency_range=(0, 4000),
                        sg_ch=True, circle_radius=5,
                        compression='gzip', compression_opts=4):
    """
    Convert a directory of .npz files to a single HDF5 file.
    
    Creates datasets for:
    - channels_100: Pre-computed 100-channel uniform channels
    - channels_400: Pre-computed 400-channel uniform channels
    - targets: Pre-computed target images
    - filenames: Original filenames for reference
    - atom_pos: Original atom positions
    - atomic_numbers: Original atomic numbers
    
    Args:
        input_dir: Directory containing .npz files
        output_path: Path to output .h5 file
        frequency_range: Tuple of (min_freq, max_freq)
        sg_ch: Single channel target (True) or multi-channel (False)
        circle_radius: Radius for circular masks
        compression: HDF5 compression type ('gzip', 'lzf', or None)
        compression_opts: Compression level (1-9 for gzip)
    """
    # Get all npz files
    npz_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
    n_samples = len(npz_files)
    
    if n_samples == 0:
        print(f"Error: No .npz files found in {input_dir}")
        return
    
    print("=" * 70)
    print("NPZ to HDF5 Converter")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output file:      {output_path}")
    print(f"Samples found:    {n_samples}")
    print(f"Settings:")
    print(f"  - Frequency range: {frequency_range}")
    print(f"  - Single channel:  {sg_ch}")
    print(f"  - Circle radius:   {circle_radius}")
    print(f"  - Compression:     {compression} (level {compression_opts})")
    print("=" * 70)
    
    # Determine shapes from first sample
    with np.load(npz_files[0]) as data:
        spectrums = data['spectrums']
        grid_size = spectrums.shape[0]  # Usually 64
    
    target_channels = 1 if sg_ch else 4  # H, C, N, O
    target_size = 256  # From molecule_circular_image
    
    print(f"\nDataset shapes:")
    print(f"  - channels_100: ({n_samples}, {grid_size}, {grid_size}, 100)")
    print(f"  - channels_400: ({n_samples}, {grid_size}, {grid_size}, 400)")
    print(f"  - targets:      ({n_samples}, {target_channels}, {target_size}, {target_size})")
    print()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Create datasets with chunking for efficient access
        channels_100_ds = hf.create_dataset(
            'channels_100',
            shape=(n_samples, grid_size, grid_size, 100),
            dtype=np.float32,
            chunks=(1, grid_size, grid_size, 100),
            compression=compression,
            compression_opts=compression_opts if compression == 'gzip' else None
        )
        
        channels_400_ds = hf.create_dataset(
            'channels_400',
            shape=(n_samples, grid_size, grid_size, 400),
            dtype=np.float32,
            chunks=(1, grid_size, grid_size, 400),
            compression=compression,
            compression_opts=compression_opts if compression == 'gzip' else None
        )
        
        targets_ds = hf.create_dataset(
            'targets',
            shape=(n_samples, target_channels, target_size, target_size),
            dtype=np.float32,
            chunks=(1, target_channels, target_size, target_size),
            compression=compression,
            compression_opts=compression_opts if compression == 'gzip' else None
        )
        
        # Store filenames for reference
        filenames = [os.path.basename(f) for f in npz_files]
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('filenames', data=filenames, dtype=dt)
        
        # Create variable-length datasets for atom data
        # (different molecules have different numbers of atoms)
        atom_pos_group = hf.create_group('atom_positions')
        atomic_nums_group = hf.create_group('atomic_numbers')
        
        # Store metadata
        hf.attrs['n_samples'] = n_samples
        hf.attrs['grid_size'] = grid_size
        hf.attrs['target_size'] = target_size
        hf.attrs['target_channels'] = target_channels
        hf.attrs['sg_ch'] = sg_ch
        hf.attrs['circle_radius'] = circle_radius
        hf.attrs['frequency_range_min'] = frequency_range[0]
        hf.attrs['frequency_range_max'] = frequency_range[1]
        hf.attrs['source_directory'] = input_dir
        
        # Process each sample
        errors = []
        for i, npz_path in enumerate(tqdm(npz_files, desc="Converting")):
            filename = os.path.splitext(os.path.basename(npz_path))[0]
            
            try:
                # Load original data
                with np.load(npz_path) as data:
                    atom_pos = data['atom_pos']
                    atomic_numbers = data['atomic_numbers']
                    frequencies = data['frequencies']
                    spectrums = data['spectrums']
                
                # Filter by frequency range
                mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
                filtered_frequencies = frequencies[mask]
                filtered_spectrums = spectrums[:, :, mask]
                
                # 1. Compute uniform_channels for both 100 and 400 channels
                channels_100 = uniform_channels(filtered_spectrums, filtered_frequencies, num_channels=100)
                channels_400 = uniform_channels(filtered_spectrums, filtered_frequencies, num_channels=400)
                
                # 2. Compute target_image
                target_image = compute_target_image(atom_pos, atomic_numbers, sg_ch=sg_ch, circle_radius=circle_radius)
                
                # Store in HDF5
                channels_100_ds[i] = channels_100
                channels_400_ds[i] = channels_400
                targets_ds[i] = target_image.astype(np.float32)
                
                # Store atom data (variable length per molecule)
                atom_pos_group.create_dataset(str(i), data=atom_pos, compression=compression)
                atomic_nums_group.create_dataset(str(i), data=atomic_numbers, compression=compression)
                
            except Exception as e:
                errors.append(f"{filename}: {e}")
                # Fill with zeros on error
                channels_100_ds[i] = np.zeros((grid_size, grid_size, 100), dtype=np.float32)
                channels_400_ds[i] = np.zeros((grid_size, grid_size, 400), dtype=np.float32)
                targets_ds[i] = np.zeros((target_channels, target_size, target_size), dtype=np.float32)
    
    # Summary
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Output file:    {output_path}")
    print(f"File size:      {file_size_gb:.2f} GB")
    print(f"Total samples:  {n_samples}")
    print(f"Errors:         {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ files to HDF5 format with pre-computed channels and targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  python npz_to_hdf5_converter.py --input_dir ./train --output ./train.h5
  
  # With custom settings:
  python npz_to_hdf5_converter.py \\
      --input_dir /data/planar_npz/train \\
      --output /data/planar_hdf5/train.h5 \\
      --circle_radius 5 \\
      --sg_ch \\
      --compression gzip \\
      --compression_level 4
  
  # Multi-channel targets:
  python npz_to_hdf5_converter.py --input_dir ./train --output ./train.h5 --no-sg_ch
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .npz files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .h5 file path')
    parser.add_argument('--freq_min', type=int, default=0,
                        help='Minimum frequency (default: 0)')
    parser.add_argument('--freq_max', type=int, default=4000,
                        help='Maximum frequency (default: 4000)')
    parser.add_argument('--circle_radius', type=int, default=5,
                        help='Circle radius for target masks (default: 5)')
    parser.add_argument('--sg_ch', action='store_true', default=True,
                        help='Single channel target (default: True)')
    parser.add_argument('--no-sg_ch', action='store_false', dest='sg_ch',
                        help='Multi-channel target (4 channels: H, C, N, O)')
    parser.add_argument('--compression', type=str, default='gzip',
                        choices=['gzip', 'lzf', 'none'],
                        help='Compression type (default: gzip)')
    parser.add_argument('--compression_level', type=int, default=4,
                        help='Compression level for gzip, 1-9 (default: 4)')
    
    args = parser.parse_args()
    
    # Handle compression
    compression = None if args.compression == 'none' else args.compression
    
    convert_npz_to_hdf5(
        input_dir=args.input_dir,
        output_path=args.output,
        frequency_range=(args.freq_min, args.freq_max),
        sg_ch=args.sg_ch,
        circle_radius=args.circle_radius,
        compression=compression,
        compression_opts=args.compression_level
    )


if __name__ == '__main__':
    main()
