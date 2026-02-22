"""
Visualization Script

Visualize:
1. Shifu and Apprentice sequences side by side
2. Ground truth vs Model prediction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from model import SiameseLSTM
from dataset import MotionPairDataset


def visualize_pair(
    control_seq: np.ndarray,
    test_seq: np.ndarray,
    true_label: float,
    pred_score: float,
    sample_idx: int,
    save_path: str = None
):
    """
    Visualize a single pair of sequences.

    Args:
        control_seq: [seq_len, 7] - shifu sequence
        test_seq: [seq_len, 7] - apprentice sequence
        true_label: Ground truth similarity
        pred_score: Model predicted similarity
        sample_idx: Sample index for title
        save_path: Optional path to save figure
    """
    seq_len = control_seq.shape[0]
    t = np.arange(seq_len) / 150.0  # Convert to seconds (150Hz)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Column titles
    axes[0, 0].set_title('SHIFU (Control)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('APPRENTICE (Test)', fontsize=14, fontweight='bold')

    # Row 1: Accelerometer
    for col, (seq, name) in enumerate([(control_seq, 'Shifu'), (test_seq, 'Apprentice')]):
        ax = axes[0, col]
        ax.plot(t, seq[:, 0], label='ax', alpha=0.8)
        ax.plot(t, seq[:, 1], label='ay', alpha=0.8)
        ax.plot(t, seq[:, 2], label='az', alpha=0.8)
        ax.set_ylabel('Accelerometer (g)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t[-1])

    # Row 2: Gyroscope
    for col, (seq, name) in enumerate([(control_seq, 'Shifu'), (test_seq, 'Apprentice')]):
        ax = axes[1, col]
        ax.plot(t, seq[:, 3], label='gx', alpha=0.8)
        ax.plot(t, seq[:, 4], label='gy', alpha=0.8)
        ax.plot(t, seq[:, 5], label='gz', alpha=0.8)
        ax.set_ylabel('Gyroscope (deg/s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t[-1])

    # Row 3: Flex sensor
    for col, (seq, name) in enumerate([(control_seq, 'Shifu'), (test_seq, 'Apprentice')]):
        ax = axes[2, col]
        ax.plot(t, seq[:, 6], label='flex', color='purple', linewidth=2)
        ax.set_ylabel('Flex (degrees)')
        ax.set_xlabel('Time (seconds)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t[-1])

    # Title with scores
    error = abs(pred_score - true_label)

    if pred_score >= 0.8:
        feedback = 'EXCELLENT'
        color = 'green'
    elif pred_score >= 0.6:
        feedback = 'GOOD'
        color = 'green'
    elif pred_score >= 0.4:
        feedback = 'NEEDS IMPROVEMENT'
        color = 'orange'
    else:
        feedback = 'OUT OF SYNC'
        color = 'red'

    fig.suptitle(
        f'Sample {sample_idx}  |  '
        f'Ground Truth: {true_label:.2f}  |  '
        f'Model Prediction: {pred_score:.4f}  |  '
        f'Error: {error:.4f}  |  '
        f'Feedback: {feedback}',
        fontsize=12,
        fontweight='bold',
        color=color
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def visualize_overlay(
    control_seq: np.ndarray,
    test_seq: np.ndarray,
    true_label: float,
    pred_score: float,
    sample_idx: int,
    save_path: str = None
):
    """
    Visualize sequences overlaid on same plot for direct comparison.
    """
    seq_len = control_seq.shape[0]
    t = np.arange(seq_len) / 150.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Accelerometer overlay
    axes[0].plot(t, control_seq[:, 0], 'b-', label='Shifu ax', alpha=0.7)
    axes[0].plot(t, test_seq[:, 0], 'r--', label='Apprentice ax', alpha=0.7)
    axes[0].plot(t, control_seq[:, 2], 'b-', label='Shifu az', alpha=0.5)
    axes[0].plot(t, test_seq[:, 2], 'r--', label='Apprentice az', alpha=0.5)
    axes[0].set_ylabel('Accelerometer (g)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Accelerometer Comparison')

    # Gyroscope overlay
    axes[1].plot(t, control_seq[:, 3], 'b-', label='Shifu gx', linewidth=2)
    axes[1].plot(t, test_seq[:, 3], 'r--', label='Apprentice gx', linewidth=2)
    axes[1].set_ylabel('Gyroscope X (deg/s)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Gyroscope X Comparison (Primary Movement Signal)')

    # Flex sensor overlay
    axes[2].plot(t, control_seq[:, 6], 'b-', label='Shifu flex', linewidth=2)
    axes[2].plot(t, test_seq[:, 6], 'r--', label='Apprentice flex', linewidth=2)
    axes[2].set_ylabel('Flex (degrees)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Flex Sensor Comparison')

    # Title
    if pred_score >= 0.6:
        color = 'green'
    elif pred_score >= 0.4:
        color = 'orange'
    else:
        color = 'red'

    fig.suptitle(
        f'Sample {sample_idx}  |  '
        f'Ground Truth: {true_label:.2f}  |  '
        f'Prediction: {pred_score:.4f}',
        fontsize=14,
        fontweight='bold',
        color=color
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def visualize_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str = None
):
    """
    Visualize overall model performance summary.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter plot: predictions vs labels
    axes[0].scatter(labels, predictions, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect')
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Model Prediction')
    axes[0].set_title('Predictions vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    # Error distribution
    errors = predictions - labels
    axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Error Distribution (MAE: {np.mean(np.abs(errors)):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-sample comparison
    indices = np.arange(len(labels))
    width = 0.35
    axes[2].bar(indices - width/2, labels, width, label='Ground Truth', alpha=0.7)
    axes[2].bar(indices + width/2, predictions, width, label='Prediction', alpha=0.7)
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Similarity Score')
    axes[2].set_title('Sample-by-Sample Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--checkpoint', type=str, default='trained_model.pth')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    print("=" * 60)
    print("Visualization")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseLSTM()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Load test data
    print(f"Loading data from {args.data_dir}/test/...")
    dataset = MotionPairDataset(args.data_dir, 'test')

    control_seqs = dataset.control_seqs
    test_seqs = dataset.test_seqs
    labels = dataset.labels

    # Get predictions
    print("Running inference...")
    with torch.no_grad():
        ctrl_tensor = torch.from_numpy(control_seqs).float().to(device)
        test_tensor = torch.from_numpy(test_seqs).float().to(device)
        predictions = model(ctrl_tensor, test_tensor).cpu().numpy().flatten()

    # Visualize summary
    print("\nGenerating summary visualization...")
    visualize_summary(
        labels, predictions,
        save_path=output_dir / 'summary.png'
    )

    # Visualize individual samples
    print(f"\nGenerating individual sample visualizations ({args.n_samples} samples)...")

    # Select diverse samples (mix of good/bad predictions)
    errors = np.abs(predictions - labels)

    # Get samples with different characteristics
    indices_to_show = []

    # Best predictions (lowest error)
    best_idx = np.argsort(errors)[:2]
    indices_to_show.extend(best_idx)

    # Worst predictions (highest error)
    worst_idx = np.argsort(errors)[-2:]
    indices_to_show.extend(worst_idx)

    # Random additional samples
    remaining = min(args.n_samples - len(indices_to_show), len(labels) - len(indices_to_show))
    if remaining > 0:
        available = [i for i in range(len(labels)) if i not in indices_to_show]
        random_idx = np.random.choice(available, remaining, replace=False)
        indices_to_show.extend(random_idx)

    indices_to_show = sorted(set(indices_to_show))[:args.n_samples]

    for idx in indices_to_show:
        # Side-by-side view
        visualize_pair(
            control_seqs[idx],
            test_seqs[idx],
            labels[idx],
            predictions[idx],
            idx,
            save_path=output_dir / f'sample_{idx:02d}_sidebyside.png'
        )

        # Overlay view
        visualize_overlay(
            control_seqs[idx],
            test_seqs[idx],
            labels[idx],
            predictions[idx],
            idx,
            save_path=output_dir / f'sample_{idx:02d}_overlay.png'
        )

    print("\n" + "=" * 60)
    print(f"Visualizations saved to {output_dir}/")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - summary.png (overall performance)")
    for idx in indices_to_show:
        print(f"  - sample_{idx:02d}_sidebyside.png")
        print(f"  - sample_{idx:02d}_overlay.png")


if __name__ == '__main__':
    main()
