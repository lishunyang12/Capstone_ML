"""
Generate test vectors from PyTorch model for HLS C simulation verification.

Creates test cases with inputs and expected outputs as text files that
the C testbench can read for bit-accurate verification.

Usage:
    python generate_test_vectors.py --checkpoint trained_model.pth --output-dir hls/test_vectors
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from model import SiameseLSTM


def save_sequence(path: Path, seq: np.ndarray):
    """Save a [seq_len, 7] sequence as a flat text file (one float per line)."""
    flat = seq.flatten()
    with open(path, 'w') as f:
        f.write(f"{seq.shape[0]}\n")  # first line: sequence length
        for v in flat:
            f.write(f"{v:.8f}\n")


def save_scalar(path: Path, value: float):
    """Save a single scalar value."""
    with open(path, 'w') as f:
        f.write(f"{value:.8f}\n")


def save_embedding(path: Path, emb: np.ndarray):
    """Save embedding vector (one float per line)."""
    with open(path, 'w') as f:
        for v in emb.flatten():
            f.write(f"{v:.8f}\n")


def generate_test_cases(model, output_dir: Path):
    """Generate 5 test cases covering different scenarios."""
    torch.manual_seed(42)
    np.random.seed(42)

    test_cases = []

    # Test 1: Identical sequences (should give high similarity)
    seq = torch.randn(1, 100, 7)
    test_cases.append(("identical", seq, seq.clone(), 100, 100))

    # Test 2: Random different sequences (similarity varies)
    seq1 = torch.randn(1, 200, 7)
    seq2 = torch.randn(1, 200, 7)
    test_cases.append(("random", seq1, seq2, 200, 200))

    # Test 3: Short sequences (minimum practical length)
    seq1 = torch.randn(1, 10, 7)
    seq2 = torch.randn(1, 10, 7)
    test_cases.append(("short", seq1, seq2, 10, 10))

    # Test 4: Different lengths (tests variable-length handling)
    seq1 = torch.randn(1, 150, 7)
    seq2 = torch.randn(1, 80, 7)
    test_cases.append(("mixed_length", seq1, seq2, 150, 80))

    # Test 5: Maximum length (stress test)
    seq1 = torch.randn(1, 450, 7)
    seq2 = torch.randn(1, 450, 7)
    test_cases.append(("max_length", seq1, seq2, 450, 450))

    model.eval()
    with torch.no_grad():
        for name, s1, s2, len1, len2 in test_cases:
            # Get model output
            score = model(s1, s2).item()

            # Get intermediate embeddings for debug
            emb1 = model.encode(s1).numpy().flatten()
            emb2 = model.encode(s2).numpy().flatten()

            # Save inputs
            save_sequence(output_dir / f"{name}_seq1.txt", s1.numpy()[0])
            save_sequence(output_dir / f"{name}_seq2.txt", s2.numpy()[0])

            # Save expected output
            save_scalar(output_dir / f"{name}_expected.txt", score)

            # Save debug embeddings
            save_embedding(output_dir / f"{name}_emb1.txt", emb1)
            save_embedding(output_dir / f"{name}_emb2.txt", emb2)

            print(f"  {name}: len1={len1}, len2={len2}, score={score:.6f}")

    # Save test manifest
    with open(output_dir / "manifest.txt", 'w') as f:
        f.write(f"{len(test_cases)}\n")
        for name, _, _, _, _ in test_cases:
            f.write(f"{name}\n")

    print(f"\n  {len(test_cases)} test cases saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate HLS test vectors")
    parser.add_argument("--checkpoint", type=str, default="trained_model.pth")
    parser.add_argument("--output-dir", type=str, default="hls/test_vectors")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = SiameseLSTM()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Generating test vectors...")
    generate_test_cases(model, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
