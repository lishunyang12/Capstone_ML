"""
Export PyTorch Siamese LSTM weights to C header files for Vitis HLS.

Produces:
    hls_weights/
        weights_lstm.h       - LSTM gate weights and biases
        weights_projection.h - Projection layer
        weights_similarity.h - Similarity head (2 FC layers)
        model_params.h       - Dimensions and constants

Usage (from project root):
    python fpga/export_weights.py --checkpoint trained_model.pth --output-dir fpga/hls_weights
"""

import torch
import numpy as np
import argparse
from pathlib import Path


def format_array_1d(name: str, values: np.ndarray, dtype: str = "float") -> str:
    """Format a 1D array as a C const array."""
    n = len(values)
    lines = [f"const {dtype} {name}[{n}] = {{"]
    # 8 values per line for readability
    for i in range(0, n, 8):
        chunk = values[i:i+8]
        row = ", ".join(f"{v: .8f}f" for v in chunk)
        if i + 8 < n:
            row += ","
        lines.append(f"    {row}")
    lines.append("};")
    return "\n".join(lines)


def format_array_2d(name: str, values: np.ndarray, dtype: str = "float") -> str:
    """Format a 2D array as a flattened C const array (row-major)."""
    rows, cols = values.shape
    flat = values.flatten()
    lines = [f"// Shape: [{rows}][{cols}], stored row-major"]
    lines.append(f"const {dtype} {name}[{rows * cols}] = {{")
    for r in range(rows):
        row_vals = values[r]
        row_str = ", ".join(f"{v: .8f}f" for v in row_vals)
        if r < rows - 1:
            row_str += ","
        lines.append(f"    /* row {r:3d} */ {row_str}")
    lines.append("};")
    return "\n".join(lines)


def export_lstm_weights(state_dict: dict, output_dir: Path):
    """
    Export LSTM weights.

    PyTorch LSTM gate order: [input_gate, forget_gate, cell_gate, output_gate]
    Each gate has hidden_dim=32 rows.

    weight_ih: [4*hidden, input_dim]  = [128, 7]
    weight_hh: [4*hidden, hidden_dim] = [128, 32]
    bias_ih:   [4*hidden]             = [128]
    bias_hh:   [4*hidden]             = [128]

    For HLS inference, we combine: bias = bias_ih + bias_hh
    """
    weight_ih = state_dict["lstm.weight_ih_l0"].numpy()  # [128, 7]
    weight_hh = state_dict["lstm.weight_hh_l0"].numpy()  # [128, 32]
    bias_ih = state_dict["lstm.bias_ih_l0"].numpy()      # [128]
    bias_hh = state_dict["lstm.bias_hh_l0"].numpy()      # [128]

    # Combine biases for inference
    bias = bias_ih + bias_hh  # [128]

    hidden_dim = 32
    input_dim = 7

    # Split into individual gates for clarity
    # PyTorch order: i, f, g, o (each is [32, ...])
    gate_names = ["i", "f", "g", "o"]

    lines = []
    lines.append("#ifndef WEIGHTS_LSTM_H")
    lines.append("#define WEIGHTS_LSTM_H")
    lines.append("")
    lines.append("// LSTM weights exported from PyTorch")
    lines.append("// Gate order: i=input, f=forget, g=cell, o=output")
    lines.append(f"// input_dim={input_dim}, hidden_dim={hidden_dim}")
    lines.append("")

    # Export full weight matrices (used in matrix multiply)
    lines.append("// --- Input weights: W_ih [4*hidden_dim, input_dim] = [128, 7] ---")
    lines.append(format_array_2d("lstm_weight_ih", weight_ih))
    lines.append("")

    lines.append("// --- Hidden weights: W_hh [4*hidden_dim, hidden_dim] = [128, 32] ---")
    lines.append(format_array_2d("lstm_weight_hh", weight_hh))
    lines.append("")

    lines.append("// --- Combined bias: b_ih + b_hh [4*hidden_dim] = [128] ---")
    lines.append(format_array_1d("lstm_bias", bias))
    lines.append("")

    # Also export per-gate views for convenience
    for g_idx, g_name in enumerate(gate_names):
        start = g_idx * hidden_dim
        end = start + hidden_dim

        lines.append(f"// --- Gate '{g_name}' input weights [{hidden_dim}, {input_dim}] ---")
        lines.append(format_array_2d(f"lstm_w_ih_{g_name}", weight_ih[start:end]))
        lines.append("")

        lines.append(f"// --- Gate '{g_name}' hidden weights [{hidden_dim}, {hidden_dim}] ---")
        lines.append(format_array_2d(f"lstm_w_hh_{g_name}", weight_hh[start:end]))
        lines.append("")

        lines.append(f"// --- Gate '{g_name}' bias [{hidden_dim}] ---")
        lines.append(format_array_1d(f"lstm_b_{g_name}", bias[start:end]))
        lines.append("")

    lines.append("#endif // WEIGHTS_LSTM_H")

    path = output_dir / "weights_lstm.h"
    path.write_text("\n".join(lines))
    print(f"  Written: {path} ({path.stat().st_size:,} bytes)")


def export_projection_weights(state_dict: dict, output_dir: Path):
    """Export projection layer: Linear(32, 16)."""
    weight = state_dict["projection.weight"].numpy()  # [16, 32]
    bias = state_dict["projection.bias"].numpy()       # [16]

    lines = []
    lines.append("#ifndef WEIGHTS_PROJECTION_H")
    lines.append("#define WEIGHTS_PROJECTION_H")
    lines.append("")
    lines.append("// Projection layer: Linear(hidden_dim=32, embedding_dim=16)")
    lines.append("")
    lines.append(format_array_2d("proj_weight", weight))
    lines.append("")
    lines.append(format_array_1d("proj_bias", bias))
    lines.append("")
    lines.append("#endif // WEIGHTS_PROJECTION_H")

    path = output_dir / "weights_projection.h"
    path.write_text("\n".join(lines))
    print(f"  Written: {path} ({path.stat().st_size:,} bytes)")


def export_similarity_weights(state_dict: dict, output_dir: Path):
    """
    Export similarity head:
        Linear(48, 16) -> ReLU -> Dropout -> Linear(16, 1) -> Sigmoid
    Dropout is identity at inference, so we skip it.
    """
    fc1_weight = state_dict["similarity_head.0.weight"].numpy()  # [16, 48]
    fc1_bias = state_dict["similarity_head.0.bias"].numpy()       # [16]
    fc2_weight = state_dict["similarity_head.3.weight"].numpy()  # [1, 16]
    fc2_bias = state_dict["similarity_head.3.bias"].numpy()       # [1]

    lines = []
    lines.append("#ifndef WEIGHTS_SIMILARITY_H")
    lines.append("#define WEIGHTS_SIMILARITY_H")
    lines.append("")
    lines.append("// Similarity head")
    lines.append("// FC1: Linear(48, 16) -> ReLU")
    lines.append("// FC2: Linear(16, 1) -> Sigmoid")
    lines.append("")
    lines.append("// --- FC1 weights [16, 48] ---")
    lines.append(format_array_2d("sim_fc1_weight", fc1_weight))
    lines.append("")
    lines.append(format_array_1d("sim_fc1_bias", fc1_bias))
    lines.append("")
    lines.append("// --- FC2 weights [1, 16] ---")
    lines.append(format_array_2d("sim_fc2_weight", fc2_weight))
    lines.append("")
    lines.append(format_array_1d("sim_fc2_bias", fc2_bias))
    lines.append("")
    lines.append("#endif // WEIGHTS_SIMILARITY_H")

    path = output_dir / "weights_similarity.h"
    path.write_text("\n".join(lines))
    print(f"  Written: {path} ({path.stat().st_size:,} bytes)")


def export_model_params(output_dir: Path):
    """Export model dimension constants."""
    lines = []
    lines.append("#ifndef MODEL_PARAMS_H")
    lines.append("#define MODEL_PARAMS_H")
    lines.append("")
    lines.append("// Model architecture constants")
    lines.append("#define INPUT_DIM       7    // ax, ay, az, gx, gy, gz, flex")
    lines.append("#define HIDDEN_DIM     32    // LSTM hidden size")
    lines.append("#define EMBEDDING_DIM  16    // Projection output size")
    lines.append("#define SIM_FC1_OUT    16    // Similarity head FC1 output")
    lines.append("#define SIM_FC2_OUT     1    // Final output (similarity score)")
    lines.append("#define NUM_GATES       4    // LSTM gates: i, f, g, o")
    lines.append("")
    lines.append("// Input constants")
    lines.append("#define SEQ_LEN       450    // 3 seconds at 150Hz")
    lines.append("#define MIN_SEQ_LEN   150    // 1 second minimum")
    lines.append("#define SAMPLING_HZ   150    // Sensor sampling rate")
    lines.append("")
    lines.append("// Derived constants")
    lines.append("#define LSTM_GATE_SIZE (NUM_GATES * HIDDEN_DIM)  // 128")
    lines.append("#define SIM_INPUT_DIM  (EMBEDDING_DIM * 3)       // 48 (emb1 + emb2 + |diff|)")
    lines.append("")
    lines.append("#endif // MODEL_PARAMS_H")

    path = output_dir / "model_params.h"
    path.write_text("\n".join(lines))
    print(f"  Written: {path} ({path.stat().st_size:,} bytes)")


def verify_export(state_dict: dict, output_dir: Path):
    """Quick sanity check: re-read exported values and compare to originals."""
    import re

    print("\n  Verification (spot-checking first value of each parameter):")
    checks = [
        ("lstm.weight_ih_l0", "weights_lstm.h", "lstm_weight_ih"),
        ("projection.weight", "weights_projection.h", "proj_weight"),
        ("similarity_head.0.weight", "weights_similarity.h", "sim_fc1_weight"),
    ]

    for param_name, filename, c_name in checks:
        original_val = state_dict[param_name].numpy().flatten()[0]
        content = (output_dir / filename).read_text()

        # Find first float value in the array
        pattern = rf"{c_name}\[.*?\]\s*=\s*\{{\s*/\*.*?\*/\s*([\-\d.]+)f"
        match = re.search(pattern, content)
        if match:
            exported_val = float(match.group(1))
            match_ok = abs(original_val - exported_val) < 1e-6
            status = "OK" if match_ok else "MISMATCH"
            print(f"    {param_name}: pytorch={original_val:.8f} export={exported_val:.8f} [{status}]")
        else:
            print(f"    {param_name}: could not parse (check manually)")


def main():
    parser = argparse.ArgumentParser(description="Export model weights to C headers")
    parser.add_argument("--checkpoint", type=str, default="trained_model.pth")
    parser.add_argument("--output-dir", type=str, default="hls_weights")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")

    print(f"\nExporting to: {output_dir}/")

    export_model_params(output_dir)
    export_lstm_weights(state_dict, output_dir)
    export_projection_weights(state_dict, output_dir)
    export_similarity_weights(state_dict, output_dir)

    verify_export(state_dict, output_dir)

    # Summary
    total_params = sum(p.numel() for p in state_dict.values())
    total_bytes = total_params * 4  # float32
    print(f"\n  Total: {total_params:,} parameters ({total_bytes:,} bytes as float32)")
    print(f"\n  Files:")
    for f in sorted(output_dir.glob("*.h")):
        print(f"    {f.name}")

    print("\nDone. Include these headers in your Vitis HLS project.")


if __name__ == "__main__":
    main()
