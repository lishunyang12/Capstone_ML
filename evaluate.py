"""
Evaluation Script - Compare hardcoded baselines vs ML model

Usage:
    python evaluate.py --data-dir data --checkpoint trained_model.pth
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import correlate

from model import SiameseLSTM
from dataset import MotionPairDataset


# =============================================================================
# Hardcoded Baselines
# =============================================================================

def baseline_angle_diff(control: np.ndarray, test: np.ndarray) -> float:
    """Direct sensor value difference."""
    diff = np.abs(control - test)
    weights = np.array([1/2, 1/2, 1/2, 1/500, 1/500, 1/500, 1/180])
    weighted_diff = diff * weights
    mean_diff = np.mean(weighted_diff)
    return float(np.clip(np.exp(-mean_diff * 5), 0, 1))


def baseline_cross_correlation(control: np.ndarray, test: np.ndarray) -> float:
    """Cross-correlation on gyroscope signal."""
    ctrl_signal = control[:, 3]
    test_signal = test[:, 3]

    ctrl_norm = (ctrl_signal - ctrl_signal.mean()) / (ctrl_signal.std() + 1e-8)
    test_norm = (test_signal - test_signal.mean()) / (test_signal.std() + 1e-8)

    correlation = correlate(ctrl_norm, test_norm, mode='same')
    correlation = correlation / len(ctrl_signal)
    max_corr = np.max(correlation)

    return float(np.clip((max_corr + 1) / 2, 0, 1))


def baseline_dtw_simple(control: np.ndarray, test: np.ndarray, window: int = 50) -> float:
    """Simplified DTW-like distance."""
    seq_len = control.shape[0]
    total_dist = 0.0

    for i in range(seq_len):
        start = max(0, i - window)
        end = min(seq_len, i + window)

        min_dist = float('inf')
        for j in range(start, end):
            dist = np.linalg.norm(control[i] - test[j])
            min_dist = min(min_dist, dist)

        total_dist += min_dist

    avg_dist = total_dist / seq_len
    return float(np.clip(np.exp(-avg_dist * 0.5), 0, 1))


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_baselines(control_seqs, test_seqs, labels):
    """Evaluate all hardcoded baselines."""
    baselines = {
        'angle_diff': baseline_angle_diff,
        'cross_corr': baseline_cross_correlation,
        'dtw_simple': baseline_dtw_simple
    }

    results = {}

    for name, func in baselines.items():
        print(f"  Evaluating {name}...")
        preds = [func(control_seqs[i], test_seqs[i]) for i in range(len(labels))]
        preds = np.array(preds)

        mae = np.mean(np.abs(preds - labels))
        acc = np.mean((preds > 0.5) == (labels > 0.5))

        results[name] = {'mae': mae, 'accuracy': acc, 'predictions': preds}

    return results


def evaluate_ml(model, control_seqs, test_seqs, labels, device):
    """Evaluate ML model."""
    model.eval()

    ctrl_tensor = torch.from_numpy(control_seqs).float().to(device)
    test_tensor = torch.from_numpy(test_seqs).float().to(device)

    with torch.no_grad():
        preds = model(ctrl_tensor, test_tensor).cpu().numpy().flatten()

    mae = np.mean(np.abs(preds - labels))
    acc = np.mean((preds > 0.5) == (labels > 0.5))

    return {'mae': mae, 'accuracy': acc, 'predictions': preds}


def main():
    parser = argparse.ArgumentParser(description='Evaluate model vs baselines')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--checkpoint', type=str, default='trained_model.pth', help='Model checkpoint')
    args = parser.parse_args()

    print("=" * 60)
    print("Evaluation: Hardcoded vs ML")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test data from {args.data_dir}/test/...")
    dataset = MotionPairDataset(args.data_dir, 'test')

    control_seqs = dataset.control_seqs
    test_seqs = dataset.test_seqs
    labels = dataset.labels

    # Evaluate hardcoded baselines
    print("\n[Hardcoded Baselines]")
    baseline_results = evaluate_baselines(control_seqs, test_seqs, labels)

    # Evaluate ML model (before loading checkpoint - random weights)
    print("\n[ML Model - Before Training]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_random = SiameseLSTM().to(device)
    ml_before = evaluate_ml(model_random, control_seqs, test_seqs, labels, device)

    # Evaluate ML model (after training)
    print("\n[ML Model - After Training]")
    if Path(args.checkpoint).exists():
        model_trained = SiameseLSTM().to(device)
        model_trained.load_state_dict(torch.load(args.checkpoint, map_location=device))
        ml_after = evaluate_ml(model_trained, control_seqs, test_seqs, labels, device)
    else:
        print(f"  Checkpoint not found: {args.checkpoint}")
        ml_after = None

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<20} {'MAE':>10} {'Accuracy':>12}")
    print("-" * 45)

    for name, res in baseline_results.items():
        print(f"{name:<20} {res['mae']:>10.4f} {res['accuracy']:>12.2%}")

    print("-" * 45)
    print(f"{'ML (before train)':<20} {ml_before['mae']:>10.4f} {ml_before['accuracy']:>12.2%}")

    if ml_after:
        print(f"{'ML (after train)':<20} {ml_after['mae']:>10.4f} {ml_after['accuracy']:>12.2%}")

    print("-" * 45)

    if ml_after:
        best_baseline = min(baseline_results.values(), key=lambda x: x['mae'])['mae']
        improvement = (best_baseline - ml_after['mae']) / best_baseline * 100
        print(f"\nML improvement over best baseline: {improvement:+.1f}%")


if __name__ == '__main__':
    main()
