"""
Dataset Loading Utilities

Expected data structure:
    data/
    ├── train/
    │   └── pairs.npz  (control_seqs, test_seqs, labels)
    ├── val/
    │   └── pairs.npz
    └── test/
        └── pairs.npz

Each .npz file contains:
    - control_seqs: [n_samples, seq_len, 7] - reference sequences
    - test_seqs: [n_samples, seq_len, 7] - test sequences
    - labels: [n_samples] - similarity scores [0, 1]

CSV format for individual sequences:
    timestamp_ms, ax, ay, az, gx, gy, gz, flex
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class MotionPairDataset(Dataset):
    """Dataset of motion sequence pairs."""

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
        """
        path = Path(data_dir) / split / 'pairs.npz'

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = np.load(path)
        self.control_seqs = data['control_seqs'].astype(np.float32)
        self.test_seqs = data['test_seqs'].astype(np.float32)
        self.labels = data['labels'].astype(np.float32)

        print(f"Loaded {split}: {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.control_seqs[idx]),
            torch.from_numpy(self.test_seqs[idx]),
            torch.tensor([self.labels[idx]], dtype=torch.float32)
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""

    train_dataset = MotionPairDataset(data_dir, 'train')
    val_dataset = MotionPairDataset(data_dir, 'val')
    test_dataset = MotionPairDataset(data_dir, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_csv_sequence(filepath: str) -> np.ndarray:
    """
    Load a single sequence from CSV.

    Expected format: timestamp_ms, ax, ay, az, gx, gy, gz, flex
    Returns: [seq_len, 7] array (excludes timestamp)
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data[:, 1:8].astype(np.float32)


def save_dataset(
    control_seqs: np.ndarray,
    test_seqs: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """
    Save a dataset split to npz format.

    Args:
        control_seqs: [n_samples, seq_len, 7]
        test_seqs: [n_samples, seq_len, 7]
        labels: [n_samples]
        output_path: Path to save (e.g., 'data/train/pairs.npz')
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        control_seqs=control_seqs.astype(np.float32),
        test_seqs=test_seqs.astype(np.float32),
        labels=labels.astype(np.float32)
    )
    print(f"Saved {len(labels)} samples to {path}")


if __name__ == '__main__':
    print("Dataset module ready.")
    print("\nExpected data structure:")
    print("  data/")
    print("  ├── train/pairs.npz")
    print("  ├── val/pairs.npz")
    print("  └── test/pairs.npz")
    print("\nEach npz contains: control_seqs, test_seqs, labels")
