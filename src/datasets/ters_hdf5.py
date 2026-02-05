"""
HDF5 Dataset for TERS data.

Fast dataset class that loads pre-computed data from HDF5 files.
Supports both 100 and 400 channel versions.

Usage:
    from src.datasets.ters_hdf5 import Ters_dataset_hdf5
    
    train_ds = Ters_dataset_hdf5(
        hdf5_path="/path/to/train.h5",
        num_channels=400,  # or 100
        t_image=transform,
        train_aug=True
    )
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class Ters_dataset_hdf5(Dataset):
    """
    Ultra-fast dataset using HDF5 format.
    
    The HDF5 file should contain:
    - channels_100: Pre-computed 100-channel uniform channels (N, H, W, 100)
    - channels_400: Pre-computed 400-channel uniform channels (N, H, W, 400)
    - targets: Pre-computed target images (N, C, H, W)
    - filenames: Original filenames for reference
    
    Benefits:
    - Single file = no filesystem overhead
    - Memory-mapped access = OS handles caching
    - Pre-computed = no CPU bottleneck
    - Chunked storage = efficient random access
    """
    
    def __init__(self, hdf5_path, num_channels=400, t_image=None, train_aug=False):
        """
        Args:
            hdf5_path: Path to HDF5 file
            num_channels: Number of channels to use (100 or 400)
            t_image: Transform to apply to images (use NormalizeVectorized!)
            train_aug: Whether to apply augmentation
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.num_channels = num_channels
        self.t_image = t_image
        self.train_aug = train_aug
        
        # Validate num_channels
        if num_channels not in [100, 400]:
            raise ValueError(f"num_channels must be 100 or 400, got {num_channels}")
        
        # Open HDF5 file (kept open for fast access)
        self.hf = h5py.File(hdf5_path, 'r')
        
        # Select appropriate channels dataset
        channels_key = f'channels_{num_channels}'
        if channels_key not in self.hf:
            raise KeyError(f"Dataset '{channels_key}' not found in HDF5 file. "
                          f"Available keys: {list(self.hf.keys())}")
        
        self.channels = self.hf[channels_key]
        self.targets = self.hf['targets']
        self.length = self.channels.shape[0]
        
        # For augmentation
        if train_aug:
            from src.transforms import AugmentTransform
            self.aug_image = AugmentTransform(gauss_std_range=(0.01, 0.1))
        
        # Print info
        print(f"Loaded HDF5 dataset: {hdf5_path}")
        print(f"  Samples:     {self.length}")
        print(f"  Channels:    {num_channels} (shape: {self.channels.shape})")
        print(f"  Targets:     {self.targets.shape}")
        print(f"  Augmentation: {train_aug}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Direct HDF5 access (FAST! Memory-mapped, no file open/close)
        channels = self.channels[idx]      # (H, W, C)
        target_image = self.targets[idx]   # (C, H, W)
        
        # Convert to tensors: (H, W, C) -> (C, H, W)
        selected_images = torch.from_numpy(channels).float().permute(2, 0, 1).contiguous()
        target_image = torch.from_numpy(target_image).float()
        
        # Apply transforms (vectorized!)
        if self.t_image:
            selected_images = self.t_image(selected_images)
        
        # Apply augmentation
        if self.train_aug:
            selected_images, target_image = self.aug_image(img=selected_images, mask=target_image)
        
        # Dummy frequencies (for compatibility with existing code)
        selected_frequencies = torch.zeros(1)
        
        return selected_images, selected_frequencies, target_image
    
    def get_filename(self, idx):
        """Get the original filename for a sample."""
        if 'filenames' in self.hf:
            return self.hf['filenames'][idx].decode() if isinstance(self.hf['filenames'][idx], bytes) else self.hf['filenames'][idx]
        return None
    
    def get_atom_data(self, idx):
        """Get atom positions and numbers for a sample."""
        if 'atom_positions' in self.hf and 'atomic_numbers' in self.hf:
            atom_pos = self.hf['atom_positions'][str(idx)][:]
            atomic_nums = self.hf['atomic_numbers'][str(idx)][:]
            return atom_pos, atomic_nums
        return None, None
    
    def get_metadata(self):
        """Get dataset metadata."""
        return dict(self.hf.attrs)
    
    def close(self):
        """Close HDF5 file when done."""
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()


class Ters_dataset_hdf5_flexible(Dataset):
    """
    Flexible HDF5 dataset that can work with any channel count.
    
    Use this if you have HDF5 files with non-standard channel counts,
    or if you want to dynamically select which channels to use.
    """
    
    def __init__(self, hdf5_path, channels_key='channels_400', t_image=None, train_aug=False):
        """
        Args:
            hdf5_path: Path to HDF5 file
            channels_key: Key for channels dataset (e.g., 'channels_100', 'channels_400')
            t_image: Transform to apply to images
            train_aug: Whether to apply augmentation
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.t_image = t_image
        self.train_aug = train_aug
        
        self.hf = h5py.File(hdf5_path, 'r')
        
        if channels_key not in self.hf:
            raise KeyError(f"Dataset '{channels_key}' not found. Available: {list(self.hf.keys())}")
        
        self.channels = self.hf[channels_key]
        self.targets = self.hf['targets']
        self.length = self.channels.shape[0]
        
        if train_aug:
            from src.transforms import AugmentTransform
            self.aug_image = AugmentTransform(gauss_std_range=(0.01, 0.1))
        
        print(f"Loaded: {hdf5_path} ({self.length} samples, {channels_key})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        channels = self.channels[idx]
        target_image = self.targets[idx]
        
        selected_images = torch.from_numpy(channels).float().permute(2, 0, 1).contiguous()
        target_image = torch.from_numpy(target_image).float()
        
        if self.t_image:
            selected_images = self.t_image(selected_images)
        
        if self.train_aug:
            selected_images, target_image = self.aug_image(img=selected_images, mask=target_image)
        
        return selected_images, torch.zeros(1), target_image
    
    def close(self):
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()
    
    def __del__(self):
        self.close()
