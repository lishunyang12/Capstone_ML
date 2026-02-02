"""
Inference Script

Usage:
    # On test dataset
    python inference.py --checkpoint trained_model.pth --data-dir data

    # On single pair of CSV files
    python inference.py --checkpoint trained_model.pth --control shifu.csv --test apprentice.csv
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from model import SiameseLSTM
from dataset import load_csv_sequence, MotionPairDataset


class TaiChiInference:
    """Inference engine for motion similarity."""

    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.model = SiameseLSTM()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")

    def predict(self, control_seq: np.ndarray, test_seq: np.ndarray) -> dict:
        """
        Predict similarity between two sequences.

        Args:
            control_seq: [seq_len, 7] - reference/shifu
            test_seq: [seq_len, 7] - test/apprentice

        Returns:
            dict with similarity_score, feedback, ar_color
        """
        if control_seq.ndim == 2:
            control_seq = control_seq[np.newaxis, ...]
        if test_seq.ndim == 2:
            test_seq = test_seq[np.newaxis, ...]

        ctrl = torch.from_numpy(control_seq).float().to(self.device)
        test = torch.from_numpy(test_seq).float().to(self.device)

        with torch.no_grad():
            score = self.model(ctrl, test).item()

        # Feedback thresholds
        if score >= 0.8:
            feedback, color = 'excellent', 'green'
        elif score >= 0.6:
            feedback, color = 'good', 'green'
        elif score >= 0.4:
            feedback, color = 'needs_improvement', 'yellow'
        else:
            feedback, color = 'out_of_sync', 'red'

        return {
            'similarity_score': round(score, 4),
            'feedback': feedback,
            'ar_color': color
        }

    def predict_batch(self, control_seqs: np.ndarray, test_seqs: np.ndarray) -> np.ndarray:
        """Batch prediction."""
        ctrl = torch.from_numpy(control_seqs).float().to(self.device)
        test = torch.from_numpy(test_seqs).float().to(self.device)

        with torch.no_grad():
            preds = self.model(ctrl, test)

        return preds.cpu().numpy().flatten()


def main():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory (for test set)')
    parser.add_argument('--control', type=str, default=None, help='Control CSV file')
    parser.add_argument('--test', type=str, default=None, help='Test CSV file')
    args = parser.parse_args()

    engine = TaiChiInference(args.checkpoint)

    if args.control and args.test:
        # Single pair inference
        print(f"\nLoading {args.control}...")
        control = load_csv_sequence(args.control)

        print(f"Loading {args.test}...")
        test = load_csv_sequence(args.test)

        result = engine.predict(control, test)

        print("\n" + "=" * 40)
        print("RESULT")
        print("=" * 40)
        print(f"Similarity: {result['similarity_score']}")
        print(f"Feedback:   {result['feedback']}")
        print(f"AR Color:   {result['ar_color']}")

    elif args.data_dir:
        # Test set inference
        print(f"\nLoading test data from {args.data_dir}/test/...")
        dataset = MotionPairDataset(args.data_dir, 'test')

        preds = engine.predict_batch(dataset.control_seqs, dataset.test_seqs)
        labels = dataset.labels

        mae = np.mean(np.abs(preds - labels))
        acc = np.mean((preds > 0.5) == (labels > 0.5))

        print(f"\nTest MAE: {mae:.4f}")
        print(f"Test Acc: {acc:.2%}")

        print("\nSample predictions:")
        print(f"{'Index':<8} {'True':>8} {'Pred':>8} {'Feedback':<15}")
        print("-" * 45)
        for i in range(min(10, len(labels))):
            res = engine.predict(dataset.control_seqs[i], dataset.test_seqs[i])
            print(f"{i:<8} {labels[i]:>8.3f} {preds[i]:>8.4f} {res['feedback']:<15}")

    else:
        print("Provide --data-dir or both --control and --test")


if __name__ == '__main__':
    main()
