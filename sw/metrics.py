"""
Metrics for Tai Chi Motion Analysis

Provides:
1. Phase Offset - Temporal alignment using cross-correlation
"""

import numpy as np
from scipy.signal import correlate
from typing import Dict


# =============================================================================
# Phase Offset (Cross-Correlation)
# =============================================================================

def compute_phase_offset(
    control_seq: np.ndarray,
    test_seq: np.ndarray,
    sampling_hz: float = 150.0
) -> float:
    """
    Compute temporal phase offset using cross-correlation.

    Args:
        control_seq: [seq_len, 7] - reference sequence
        test_seq: [seq_len, 7] - test sequence
        sampling_hz: Sampling rate in Hz

    Returns:
        phase_offset_ms: Offset in milliseconds
                        Negative = test is behind (lagging)
                        Positive = test is ahead (rushing)
    """
    # Use gyroscope X channel (most informative for arm movements)
    control_signal = control_seq[:, 3]  # gx
    test_signal = test_seq[:, 3]

    # Normalize signals
    ctrl_norm = (control_signal - control_signal.mean()) / (control_signal.std() + 1e-8)
    test_norm = (test_signal - test_signal.mean()) / (test_signal.std() + 1e-8)

    # Cross-correlation
    correlation = correlate(ctrl_norm, test_norm, mode='full')

    # Find lag at maximum correlation
    n = len(control_signal)
    lags = np.arange(-(n-1), n)

    best_lag_idx = np.argmax(correlation)
    best_lag = lags[best_lag_idx]

    # Convert lag to milliseconds
    # Positive lag means test signal is behind -> negative offset (lagging)
    # Negative lag means test signal is ahead -> positive offset (rushing)
    phase_offset_ms = best_lag * (1000.0 / sampling_hz)

    return float(phase_offset_ms)


# =============================================================================
# Combined Metrics
# =============================================================================

def compute_all_metrics(
    control_seq: np.ndarray,
    test_seq: np.ndarray,
    similarity_score: float,
    sampling_hz: float = 150.0
) -> Dict:
    """
    Compute all metrics for a sequence pair.

    Args:
        control_seq: [seq_len, 7] - reference/shifu sequence
        test_seq: [seq_len, 7] - test/apprentice sequence
        similarity_score: ML-predicted similarity [0, 1]
        sampling_hz: Sampling rate

    Returns:
        dict with metrics and feedback
    """
    # Compute phase offset
    phase_offset_ms = compute_phase_offset(control_seq, test_seq, sampling_hz)

    # Generate feedback based on similarity
    if similarity_score >= 0.8:
        sync_feedback = 'excellent'
        ar_color = 'green'
    elif similarity_score >= 0.6:
        sync_feedback = 'good'
        ar_color = 'green'
    elif similarity_score >= 0.4:
        sync_feedback = 'needs_improvement'
        ar_color = 'yellow'
    else:
        sync_feedback = 'out_of_sync'
        ar_color = 'red'

    # Timing feedback
    if abs(phase_offset_ms) < 100:
        timing_feedback = 'on_time'
    elif phase_offset_ms < 0:
        timing_feedback = 'lagging'
    else:
        timing_feedback = 'rushing'

    return {
        # Scores
        'similarity_score': round(similarity_score, 4),
        'phase_offset_ms': round(phase_offset_ms, 1),

        # Feedback
        'feedback': {
            'sync': sync_feedback,
            'timing': timing_feedback
        },

        # AR visualization
        'ar_color': ar_color,
        'show_effects': similarity_score >= 0.8
    }


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Phase Offset Test")
    print("=" * 60)

    np.random.seed(42)
    seq_len = 450
    t = np.linspace(0, 3, seq_len)

    # Generate base movement
    angle = np.sin(np.clip(t / 2.4, 0, 1) * np.pi)
    rotation = angle * (np.pi / 2)

    base_seq = np.column_stack([
        np.sin(rotation) * 0.3,
        np.zeros(seq_len),
        np.cos(rotation),
        np.gradient(rotation) * 150 * (180/np.pi),
        np.zeros(seq_len),
        np.zeros(seq_len),
        20 + angle * 40
    ]).astype(np.float32)

    # Test 1: Same sequence (no offset)
    print("\n[Test 1: Same sequence]")
    offset = compute_phase_offset(base_seq, base_seq)
    print(f"  Phase offset: {offset:.1f}ms (expected: ~0ms)")

    # Test 2: Delayed sequence (200ms)
    print("\n[Test 2: Delayed by 200ms]")
    delay_samples = int(200 * 150 / 1000)  # 30 samples
    delayed_seq = np.zeros_like(base_seq)
    delayed_seq[delay_samples:] = base_seq[:-delay_samples]
    delayed_seq[:delay_samples] = base_seq[0]
    offset = compute_phase_offset(base_seq, delayed_seq)
    print(f"  Phase offset: {offset:.1f}ms (expected: ~-200ms, lagging)")

    # Test 3: Full metrics
    print("\n[Test 3: Full metrics output]")
    result = compute_all_metrics(base_seq, delayed_seq, similarity_score=0.75)
    for k, v in result.items():
        print(f"  {k}: {v}")
