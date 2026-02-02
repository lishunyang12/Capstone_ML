"""
Generate Synthetic Dataset (for testing the pipeline)

Usage:
    python generate_synthetic.py --output data --n-train 200 --n-val 50 --n-test 50

This creates synthetic data to test the training pipeline.
Replace with real sensor data for actual training.
"""

import numpy as np
import argparse
from pathlib import Path
from dataset import save_dataset


def generate_base_movement(seq_len: int = 450) -> np.ndarray:
    """Generate a clean arm raise movement."""
    t = np.linspace(0, 3, seq_len)
    progress = np.clip(t / 2.4, 0, 1)
    angle = np.sin(progress * np.pi)
    rotation = angle * (np.pi / 2)

    ax = np.sin(rotation) * 0.3
    ay = np.zeros_like(t)
    az = np.cos(rotation)
    gx = np.gradient(rotation, 1/150) * (180 / np.pi)
    gy = np.zeros_like(t)
    gz = np.zeros_like(t)
    flex = 20 + angle * 40

    return np.column_stack([ax, ay, az, gx, gy, gz, flex]).astype(np.float32)


def generate_different_movement(seq_len: int = 450) -> np.ndarray:
    """Generate a different movement pattern."""
    t = np.linspace(0, 3, seq_len)
    progress = np.clip(t / 1.2, 0, 1)
    angle = np.sin(progress * np.pi) ** 2
    rotation = angle * (np.pi / 3)

    ax = np.sin(rotation) * 0.5
    ay = np.cos(rotation) * 0.2
    az = np.cos(rotation) * 0.8
    gx = np.gradient(rotation, 1/150) * (180 / np.pi) * 1.5
    gy = np.gradient(rotation, 1/150) * (180 / np.pi) * 0.3
    gz = np.zeros_like(t)
    flex = 30 + angle * 20

    return np.column_stack([ax, ay, az, gx, gy, gz, flex]).astype(np.float32)


def add_noise(seq: np.ndarray, level: float) -> np.ndarray:
    """Add noise to sequence."""
    noise = np.random.normal(0, level, seq.shape).astype(np.float32)
    noise[:, 3:6] *= 5
    noise[:, 6] *= 2
    return seq + noise


def change_speed(seq: np.ndarray, factor: float) -> np.ndarray:
    """Change speed by resampling."""
    n = seq.shape[0]
    new_n = int(n / factor)
    old_idx = np.arange(n)
    new_idx = np.linspace(0, n-1, new_n)

    resampled = np.zeros((new_n, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        resampled[:, i] = np.interp(new_idx, old_idx, seq[:, i])

    if new_n < n:
        padded = np.zeros_like(seq)
        padded[:new_n] = resampled
        padded[new_n:] = resampled[-1]
        return padded
    return resampled[:n]


def generate_pair() -> tuple:
    """Generate a (control, test, label) triplet."""
    control = add_noise(generate_base_movement(), 0.02)

    choice = np.random.choice(['same', 'similar', 'different_speed', 'different'])

    if choice == 'same':
        test = add_noise(control.copy(), 0.03)
        label = 1.0
    elif choice == 'similar':
        test = add_noise(control.copy(), 0.05)
        test = change_speed(test, np.random.uniform(0.9, 1.1))
        label = 0.8
    elif choice == 'different_speed':
        test = change_speed(control.copy(), np.random.choice([0.6, 0.7, 1.4, 1.5]))
        test = add_noise(test, 0.05)
        label = 0.4
    else:
        test = add_noise(generate_different_movement(), 0.05)
        label = 0.0

    return control, test, label


def generate_split(n_samples: int) -> tuple:
    """Generate a dataset split."""
    controls, tests, labels = [], [], []

    for _ in range(n_samples):
        c, t, l = generate_pair()
        controls.append(c)
        tests.append(t)
        labels.append(l)

    return (
        np.array(controls, dtype=np.float32),
        np.array(tests, dtype=np.float32),
        np.array(labels, dtype=np.float32)
    )


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--n-train', type=int, default=200, help='Training samples')
    parser.add_argument('--n-val', type=int, default=50, help='Validation samples')
    parser.add_argument('--n-test', type=int, default=50, help='Test samples')
    args = parser.parse_args()

    print("Generating synthetic dataset...")

    for split, n in [('train', args.n_train), ('val', args.n_val), ('test', args.n_test)]:
        print(f"\n{split}: {n} samples")
        controls, tests, labels = generate_split(n)
        save_dataset(controls, tests, labels, f"{args.output}/{split}/pairs.npz")
        print(f"  Labels: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
