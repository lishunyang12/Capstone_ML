"""
Verify HLS C logic by replicating it in Python (pure NumPy, no PyTorch).

This reads the exported weight headers directly and the test vectors,
then runs the exact same computation as siamese_lstm.cpp to verify
correctness before deploying to Vitis HLS.

Usage (from fpga/ directory):
    python verify_hls_logic.py
"""

import os
import numpy as np
import re
from pathlib import Path


def parse_c_array(header_text: str, name: str) -> np.ndarray:
    """Extract a const float array from C header text."""
    # Match: const float name[size] = { ... };
    pattern = rf'const float {re.escape(name)}\[(\d+)\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, header_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in header")
    size = int(match.group(1))
    body = match.group(2)
    # Extract all float literals
    values = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?=f)', body)]
    assert len(values) == size, f"{name}: expected {size} values, got {len(values)}"
    return np.array(values, dtype=np.float32)


def load_weights(weights_dir: Path):
    """Load all weights from C headers."""
    lstm_h = (weights_dir / "weights_lstm.h").read_text()
    proj_h = (weights_dir / "weights_projection.h").read_text()
    sim_h = (weights_dir / "weights_similarity.h").read_text()

    w = {}
    w['lstm_weight_ih'] = parse_c_array(lstm_h, 'lstm_weight_ih').reshape(128, 7)
    w['lstm_weight_hh'] = parse_c_array(lstm_h, 'lstm_weight_hh').reshape(128, 32)
    w['lstm_bias'] = parse_c_array(lstm_h, 'lstm_bias')  # [128]
    w['proj_weight'] = parse_c_array(proj_h, 'proj_weight').reshape(16, 32)
    w['proj_bias'] = parse_c_array(proj_h, 'proj_bias')  # [16]
    w['sim_fc1_weight'] = parse_c_array(sim_h, 'sim_fc1_weight').reshape(16, 48)
    w['sim_fc1_bias'] = parse_c_array(sim_h, 'sim_fc1_bias')  # [16]
    w['sim_fc2_weight'] = parse_c_array(sim_h, 'sim_fc2_weight').reshape(1, 16)
    w['sim_fc2_bias'] = parse_c_array(sim_h, 'sim_fc2_bias')  # [1]
    return w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def lstm_encode(seq, seq_len, weights):
    """Replicate the HLS lstm_encode function exactly."""
    W_ih = weights['lstm_weight_ih']  # [128, 7]
    W_hh = weights['lstm_weight_hh']  # [128, 32]
    bias = weights['lstm_bias']       # [128]

    h = np.zeros(32, dtype=np.float32)
    c = np.zeros(32, dtype=np.float32)

    for t in range(seq_len):
        x_t = seq[t]  # [7]

        # gates = W_ih @ x_t + W_hh @ h + bias
        gates = np.zeros(128, dtype=np.float32)
        for g in range(128):
            acc = float(bias[g])
            for j in range(7):
                acc += float(W_ih[g, j]) * float(x_t[j])
            for j in range(32):
                acc += float(W_hh[g, j]) * float(h[j])
            gates[g] = np.float32(acc)

        # Apply activations — matching C code exactly
        for i in range(32):
            ig = 1.0 / (1.0 + np.exp(-float(gates[i])))            # input gate
            fg = 1.0 / (1.0 + np.exp(-float(gates[32 + i])))       # forget gate
            gg = np.tanh(float(gates[64 + i]))                       # cell gate
            og = 1.0 / (1.0 + np.exp(-float(gates[96 + i])))       # output gate

            c[i] = np.float32(fg * float(c[i]) + ig * gg)
            h[i] = np.float32(og * np.tanh(float(c[i])))

    # Projection: Linear(32, 16)
    proj_w = weights['proj_weight']  # [16, 32]
    proj_b = weights['proj_bias']    # [16]
    embedding = np.zeros(16, dtype=np.float32)
    for i in range(16):
        acc = float(proj_b[i])
        for j in range(32):
            acc += float(proj_w[i, j]) * float(h[j])
        embedding[i] = np.float32(acc)

    return embedding


def similarity_head(emb1, emb2, weights):
    """Replicate the HLS similarity_head function exactly."""
    # Build combined: [emb1, emb2, |emb1-emb2|]
    combined = np.zeros(48, dtype=np.float32)
    for i in range(16):
        combined[i] = emb1[i]
        combined[16 + i] = emb2[i]
        diff = float(emb1[i]) - float(emb2[i])
        combined[32 + i] = np.float32(abs(diff))

    # FC1: Linear(48, 16) + ReLU
    fc1_w = weights['sim_fc1_weight']  # [16, 48]
    fc1_b = weights['sim_fc1_bias']    # [16]
    fc1_out = np.zeros(16, dtype=np.float32)
    for i in range(16):
        acc = float(fc1_b[i])
        for j in range(48):
            acc += float(fc1_w[i, j]) * float(combined[j])
        fc1_out[i] = np.float32(max(0.0, acc))  # ReLU

    # FC2: Linear(16, 1) + Sigmoid
    fc2_w = weights['sim_fc2_weight']  # [1, 16]
    fc2_b = weights['sim_fc2_bias']    # [1]
    acc = float(fc2_b[0])
    for j in range(16):
        acc += float(fc2_w[0, j]) * float(fc1_out[j])
    result = np.float32(1.0 / (1.0 + np.exp(-acc)))

    return float(result)


def read_sequence(path):
    """Read sequence file (same format as C testbench)."""
    with open(path, 'r') as f:
        seq_len = int(f.readline().strip())
        values = [float(line.strip()) for line in f if line.strip()]
    return seq_len, np.array(values, dtype=np.float32).reshape(seq_len, 7)


def read_scalar(path):
    with open(path, 'r') as f:
        return float(f.readline().strip())


def read_embedding(path):
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)


def main():
    base = Path(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = base / "hls_weights"
    test_dir = base / "hls" / "test_vectors"

    print("Loading weights from C headers...")
    weights = load_weights(weights_dir)
    print(f"  lstm_weight_ih: {weights['lstm_weight_ih'].shape}")
    print(f"  lstm_weight_hh: {weights['lstm_weight_hh'].shape}")
    print(f"  lstm_bias:      {weights['lstm_bias'].shape}")

    # Read manifest
    with open(test_dir / "manifest.txt") as f:
        num_tests = int(f.readline().strip())
        test_names = [f.readline().strip() for _ in range(num_tests)]

    print(f"\nRunning {num_tests} tests (replicating HLS C logic in Python)...\n")
    print(f"  {'Name':<15} | {'Len1':>4} {'Len2':>4} | {'PyTorch':>10} {'HLS-Py':>10} | {'Error':>10} | Status")
    print("  " + "-" * 85)

    tolerance = 1e-4
    failures = 0

    for name in test_names:
        seq1_len, seq1 = read_sequence(test_dir / f"{name}_seq1.txt")
        seq2_len, seq2 = read_sequence(test_dir / f"{name}_seq2.txt")
        expected = read_scalar(test_dir / f"{name}_expected.txt")
        expected_emb1 = read_embedding(test_dir / f"{name}_emb1.txt")
        expected_emb2 = read_embedding(test_dir / f"{name}_emb2.txt")

        # Run HLS-equivalent computation
        emb1 = lstm_encode(seq1, seq1_len, weights)
        emb2 = lstm_encode(seq2, seq2_len, weights)
        result = similarity_head(emb1, emb2, weights)

        error = abs(result - expected)
        passed = error < tolerance

        # Also check embedding accuracy
        emb1_err = np.max(np.abs(emb1 - expected_emb1))
        emb2_err = np.max(np.abs(emb2 - expected_emb2))

        status = "PASS" if passed else "FAIL"
        print(f"  {name:<15} | {seq1_len:>4} {seq2_len:>4} | {expected:>10.6f} {result:>10.6f} | {error:>10.2e} | {status}")

        if not passed:
            failures += 1
            print(f"    emb1 max err: {emb1_err:.2e}, emb2 max err: {emb2_err:.2e}")

    print()
    print("=" * 60)
    if failures == 0:
        print(f"ALL {num_tests} TESTS PASSED")
        print("HLS C logic will match PyTorch output within tolerance.")
    else:
        print(f"FAILED: {failures} / {num_tests} tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
