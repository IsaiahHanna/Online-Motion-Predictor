""""
dataset.py - Motion Window Dataset
Online Human Intent Predictor with Adaptive Learning

Wraps the .npy files produced by preprocess.py into a torch Dataset.
Supports memory-mapped loading (default) so the full train set doesn't 
need to fit into RAM all at once.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MotionWindowDataset(Dataset):
    """
    Dataset of sliding windows of 3D joint positions.

    Loads X_{split}.npy and Y_{split}.npy produced by preprocess.py

    Parameters
    ----------
    data_dir : str | Path
        Directory containing X_train.npy, Y_train.npy, etc.
    split : 'train' | 'val' | 'test'
        Which split to load
    mmap : bool
        If True (default), memory-map the .npy files. The arrays are
        read from disk on demand rather than loaded entirely into RAM.
        Set to False only if you have >8 GB RAM and want maximum
        DataLoader throughput.
    augment : bool
        If True, apply light data augmentation (train split only):
            - small Gaussian noise on joing positions (sigma = 1mm)
            - random temporal flip (mirror the time axis)
        Leave False for val/test
    noise_std : float
        Std of Gaussian noise added during augmentatino (metres).
        Default 0.001 = 1 mm, well below the ADE target of 50 mm.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        mmap: bool = True,
        augment: bool = False,
        noise_std: float = 0.001
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        
        data_dir = Path(data_dir)
        x_path   = data_dir / f"X_{split}.npy"
        y_path   = data_dir / f"Y_{split}.npy"

        if not x_path.exists():
            raise FileNotFoundError(
                f"X_{split}.npy not found in {data_dir}. "
                "Run preprocess.py first."
            )

        mmap_mode = "r" if mmap else None
        self.X = np.load(x_path, mmap_mode=mmap_mode)   # [N, W, 99]
        self.Y = np.load(y_path, mmap_mode=mmap_mode)   # [N. K, 99]

        assert self.X.shape[0] == self.Y.shape[0], (
            f"X and Y window counts differ: {self.X.shape[0]} vs {self.Y.shape[0]}"
        )

        self.augment   = augment
        self.noise_std = noise_std
        self.split     = split

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
 
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self,idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Copy from mmap array (avoids modifying the original mapping)
        x = torch.from_numpy(self.X[idx].copy())  # [W, 99]
        y = torch.from_numpy(self.Y[idx].copy())  # [K, 99]

        if self.augment:
            x, y = self._augment(x, y)
        
        return x, y
    
    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
 
    def _augment(
        self, x: torch.tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Light augmentation that preserves motion semantics"""
        # 1. Small positional noise (1 mm std) - simulates sensor noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
            y = y + torch.randn_like(y) * self.noise_std
        
        # 2. Random temporal flip with p=0.5
        #   Reverses the time axis on both x and y, then swaps roles:
        #   the reversed y becomes the new x tail, reversed x becomes y.
        #   Only valid when W == 2*K (30 == 2*15), which is our setup.
        #   Skipped otherwise to avoid shape mismatch
        W = x.shape[0]
        K = y.shape[0]
        if torch.rand(1).item() < 0.5 and W == 2 * K:
            combined = torch.cat([x, y], dim=0)    # [W+K, 99]
            combined = combined.flip(0)            # reverse time
            x = combined[:W]
            y = combined[W:]

        return x, y
    
    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def n_windows(self) -> int:
        return self.X.shape[0]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]

    @property
    def horizon(self) -> int:
        return self.Y.shape[1]

    @property 
    def n_dims(self) -> int:
        return self.X.shape[2]

    def __repr__(self) -> str:
        return (
            f"MotionWindowDataset(split='{self.split}',"
            f"n={self.n_windows:,}, "
            f"x={tuple(self.X.shape)}, "
            f"Y={tuple(self.Y.shape)}, "
            f"augment={self.augment})"
        )

# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4, 
    pin_memory: bool = True,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from preprocessed data.

    Parameters
    ----------
    data_dir      : directory with X_train.npy, T_train.npy, etc.
    batch_size    : windows per batch (256 is the defualt for test)
    num_workers   : parallel workers for data loading 
    pin_memory    : enable pinned memory for faster GPU transfer
                    (set to False if running CPU-only)
    augment_train : apply noise + temporal flip to training windows

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds  = MotionWindowDataset(data_dir, "train", augment=augment_train)
    val_ds    = MotionWindowDataset(data_dir, "val",   augment=False)
    test_ds   = MotionWindowDataset(data_dir, "test",  augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,               # keeps batch size constant, good for LR schedulers
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,     # no grad, can fit larget batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader

# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        print("     e.g.  python dataset.py Data/output")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    print(f"Loading from: {data_dir}\n")

    for split in ("train", "val", "test"):
        ds = MotionWindowDataset(data_dir, split, augment=(split == "train"))
        print(ds)
        x, y = ds[0]
        print(f"    Sample shapes: x={tuple(x.shape)}  y={tuple(y.shape)}")
        print(f"    x dtype: {x.dtype}    x range: [{x.min():.3f}, {x.max():.3f}]")
        print()

        print("Building Dataloaders...")
        train_loader, val_loader, test_loader = build_dataloaders(
            data_dir, batch_size=256, num_workers=0, pin_memory=False
        )
        xb, yb = next(iter(train_loader))
        print(f"Train batch: x={tuple(xb.shape)}  y={tuple(yb.shape)}  dtype{xb.dtype}")
        print("\nDataset self-test passed.")

