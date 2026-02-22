"""
Simulate Online Serving Environment

Simulates:
1. Two sensor streams (Shifu + Apprentice) at 150Hz
2. Real-time prediction at 10Hz
3. Live visualization of results

Usage:
    python simulate_online.py
    python simulate_online.py --scenario similar
    python simulate_online.py --scenario delayed
    python simulate_online.py --scenario different
"""

import numpy as np
import time
import threading
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional
import sys

from online_serving import OnlinePredictor


# =============================================================================
# Simulated Sensor Stream
# =============================================================================

class SimulatedSensor:
    """Simulates a sensor stream at specified Hz."""

    def __init__(self, sampling_hz: float = 150.0):
        self.sampling_hz = sampling_hz
        self.interval = 1.0 / sampling_hz
        self.running = False
        self.thread = None
        self.callback = None
        self.sequence = None
        self.current_idx = 0

    def load_sequence(self, sequence: np.ndarray):
        """Load a sequence to stream."""
        self.sequence = sequence
        self.current_idx = 0

    def set_callback(self, callback):
        """Set callback function(data) called for each reading."""
        self.callback = callback

    def _stream_loop(self):
        """Background streaming loop."""
        while self.running and self.current_idx < len(self.sequence):
            if self.callback:
                self.callback(self.sequence[self.current_idx])
            self.current_idx += 1
            time.sleep(self.interval)

        # Loop the sequence
        if self.running:
            self.current_idx = 0
            self._stream_loop()

    def start(self):
        """Start streaming."""
        if self.sequence is None:
            raise ValueError("No sequence loaded")
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)


# =============================================================================
# Movement Generators
# =============================================================================

def generate_arm_raise(duration: float = 3.0, sampling_hz: float = 150.0) -> np.ndarray:
    """Generate smooth arm raise movement."""
    n = int(duration * sampling_hz)
    t = np.linspace(0, duration, n)

    progress = np.clip(t / (duration * 0.8), 0, 1)
    angle = np.sin(progress * np.pi)
    rotation = angle * (np.pi / 2)

    ax = np.sin(rotation) * 0.3
    ay = np.zeros(n)
    az = np.cos(rotation)
    gx = np.gradient(rotation) * sampling_hz * (180 / np.pi)
    gy = np.zeros(n)
    gz = np.zeros(n)
    flex = 20 + angle * 40

    seq = np.column_stack([ax, ay, az, gx, gy, gz, flex]).astype(np.float32)
    # Add small noise
    seq += np.random.normal(0, 0.02, seq.shape).astype(np.float32)
    seq[:, 3:6] += np.random.normal(0, 1, (n, 3)).astype(np.float32)

    return seq


def generate_delayed_movement(base_seq: np.ndarray, delay_ms: float) -> np.ndarray:
    """Generate same movement but delayed."""
    delay_samples = int(delay_ms * 150 / 1000)
    delayed = np.zeros_like(base_seq)
    delayed[delay_samples:] = base_seq[:-delay_samples]
    delayed[:delay_samples] = base_seq[0]
    # Add small noise
    delayed += np.random.normal(0, 0.02, delayed.shape).astype(np.float32)
    return delayed


def generate_different_speed(base_seq: np.ndarray, speed_factor: float) -> np.ndarray:
    """Generate same movement at different speed."""
    n = base_seq.shape[0]
    new_n = int(n / speed_factor)

    old_idx = np.arange(n)
    new_idx = np.linspace(0, n - 1, new_n)

    resampled = np.zeros((new_n, base_seq.shape[1]), dtype=np.float32)
    for i in range(base_seq.shape[1]):
        resampled[:, i] = np.interp(new_idx, old_idx, base_seq[:, i])

    # Pad or truncate
    if new_n < n:
        padded = np.zeros_like(base_seq)
        padded[:new_n] = resampled
        padded[new_n:] = resampled[-1]
        return padded
    return resampled[:n]


def generate_different_movement(duration: float = 3.0, sampling_hz: float = 150.0) -> np.ndarray:
    """Generate completely different movement."""
    n = int(duration * sampling_hz)
    t = np.linspace(0, duration, n)

    # Different pattern
    progress = np.clip(t / (duration * 0.4), 0, 1)
    angle = np.sin(progress * np.pi) ** 2
    rotation = angle * (np.pi / 3)

    ax = np.sin(rotation) * 0.5 + np.random.normal(0, 0.05, n)
    ay = np.cos(rotation) * 0.2 + np.random.normal(0, 0.05, n)
    az = np.cos(rotation) * 0.8 + np.random.normal(0, 0.02, n)
    gx = np.gradient(rotation) * sampling_hz * (180 / np.pi) * 1.5
    gy = np.gradient(rotation) * sampling_hz * (180 / np.pi) * 0.3
    gz = np.random.normal(0, 2, n)
    flex = 30 + angle * 20 + np.random.normal(0, 1, n)

    return np.column_stack([ax, ay, az, gx, gy, gz, flex]).astype(np.float32)


# =============================================================================
# Console Display
# =============================================================================

class ConsoleDisplay:
    """Real-time console display for predictions."""

    def __init__(self):
        self.history = deque(maxlen=20)

    def update(self, result: dict, scenario: str):
        """Update display with new result."""
        self.history.append(result)

        # Clear screen (cross-platform)
        print("\033[2J\033[H", end="")

        # Header
        print("=" * 70)
        print("       REAL-TIME TAI CHI MOTION SYNC - ONLINE SERVING DEMO")
        print("=" * 70)
        print(f"Scenario: {scenario}")
        print(f"Press Ctrl+C to stop\n")

        # Current result
        sim = result['similarity_score']
        phase = result['phase_offset_ms']
        feedback = result['feedback']

        # Color codes
        if result['ar_color'] == 'green':
            color = '\033[92m'  # Green
        elif result['ar_color'] == 'yellow':
            color = '\033[93m'  # Yellow
        else:
            color = '\033[91m'  # Red
        reset = '\033[0m'
        bold = '\033[1m'

        print(f"{bold}CURRENT PREDICTION:{reset}")
        print("-" * 40)

        # Similarity bar
        bar_len = 30
        filled = int(sim * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"Similarity:  {color}[{bar}] {sim:.1%}{reset}")

        # Phase offset indicator
        phase_indicator = ""
        if abs(phase) < 100:
            phase_indicator = "  * ON TIME"
        elif phase < 0:
            phase_indicator = f"  < LAGGING {abs(phase):.0f}ms"
        else:
            phase_indicator = f"  > RUSHING {phase:.0f}ms"
        print(f"Timing:      {color}{phase_indicator}{reset}")

        print("-" * 40)
        print(f"{bold}Feedback:{reset} {color}{feedback['sync'].upper()}{reset}")
        print(f"AR Color: {color}[*]{reset} {result['ar_color'].upper()}")

        if result.get('show_effects'):
            print(f"\n{bold}*** MOTIVATIONAL EFFECTS TRIGGERED! ***{reset}")

        # History
        print(f"\n{bold}HISTORY (last 10):{reset}")
        print("-" * 60)
        print(f"{'Time':<10} {'Score':>10} {'Phase':>12} {'Status':<15}")
        print("-" * 60)

        for i, r in enumerate(list(self.history)[-10:]):
            t = f"{i*0.1:.1f}s ago"
            s = f"{r['similarity_score']:.3f}"
            p = f"{r['phase_offset_ms']:+.0f}ms"
            status = r['feedback']['sync']

            if r['ar_color'] == 'green':
                c = '\033[92m'
            elif r['ar_color'] == 'yellow':
                c = '\033[93m'
            else:
                c = '\033[91m'

            print(f"{t:<10} {c}{s:>10} {p:>12} {status:<15}{reset}")


# =============================================================================
# Main Simulation
# =============================================================================

def run_simulation(scenario: str = 'mixed', duration: float = 30.0):
    """
    Run the online serving simulation.

    Scenarios:
        - similar: Apprentice closely follows Shifu
        - delayed: Apprentice is 200ms behind
        - different: Apprentice does wrong movement
        - fast: Apprentice moves too fast
        - slow: Apprentice moves too slow
        - mixed: Cycles through different scenarios
    """

    print("Initializing simulation...")

    # Initialize predictor
    predictor = OnlinePredictor(
        checkpoint_path='trained_model.pth',
        window_size=450,
        min_samples=150
    )

    # Generate base movement
    base_movement = generate_arm_raise(duration=3.0)

    # Generate apprentice movements for each scenario
    scenarios = {
        'similar': base_movement + np.random.normal(0, 0.02, base_movement.shape).astype(np.float32),
        'delayed': generate_delayed_movement(base_movement, delay_ms=200),
        'different': generate_different_movement(),
        'fast': generate_different_speed(base_movement, 1.5),
        'slow': generate_different_speed(base_movement, 0.6),
    }

    # Setup sensors
    shifu_sensor = SimulatedSensor(sampling_hz=150)
    apprentice_sensor = SimulatedSensor(sampling_hz=150)

    shifu_sensor.set_callback(lambda data: predictor.add_shifu_reading(data))
    apprentice_sensor.set_callback(lambda data: predictor.add_apprentice_reading(data))

    # Console display
    display = ConsoleDisplay()

    # Current scenario
    current_scenario = scenario if scenario != 'mixed' else 'similar'
    scenario_cycle = ['similar', 'delayed', 'fast', 'slow', 'different']
    scenario_idx = 0

    def load_scenario(name):
        nonlocal current_scenario
        current_scenario = name
        shifu_sensor.stop()
        apprentice_sensor.stop()
        predictor.reset()

        shifu_sensor.load_sequence(base_movement)
        apprentice_sensor.load_sequence(scenarios[name])

        shifu_sensor.start()
        apprentice_sensor.start()

    # Load initial scenario
    load_scenario(current_scenario)

    print("\nStarting real-time simulation...")
    print("Press Ctrl+C to stop\n")
    time.sleep(1)

    start_time = time.time()
    last_scenario_change = start_time
    scenario_duration = 6.0  # Change scenario every 6 seconds in mixed mode

    try:
        while (time.time() - start_time) < duration:
            # Prediction
            result = predictor.predict()

            if result:
                display.update(result, current_scenario)

            # Change scenario in mixed mode
            if scenario == 'mixed' and (time.time() - last_scenario_change) > scenario_duration:
                scenario_idx = (scenario_idx + 1) % len(scenario_cycle)
                load_scenario(scenario_cycle[scenario_idx])
                last_scenario_change = time.time()

            time.sleep(0.1)  # 10Hz display update

    except KeyboardInterrupt:
        print("\n\nStopping simulation...")

    finally:
        shifu_sensor.stop()
        apprentice_sensor.stop()

    print("\nSimulation complete!")


def main():
    parser = argparse.ArgumentParser(description='Simulate online serving')
    parser.add_argument('--scenario', type=str, default='mixed',
                        choices=['similar', 'delayed', 'different', 'fast', 'slow', 'mixed'],
                        help='Scenario to simulate')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Simulation duration in seconds')
    args = parser.parse_args()

    print("=" * 70)
    print("TAI CHI MOTION SYNC - ONLINE SERVING SIMULATION")
    print("=" * 70)
    print(f"\nScenario: {args.scenario}")
    print(f"Duration: {args.duration}s")
    print("\nScenario descriptions:")
    print("  similar   - Apprentice closely follows Shifu (expect: green)")
    print("  delayed   - Apprentice is 200ms behind (expect: green, lagging)")
    print("  different - Apprentice does wrong movement (expect: red)")
    print("  fast      - Apprentice moves 1.5x faster (expect: yellow/red)")
    print("  slow      - Apprentice moves 0.6x slower (expect: yellow/red)")
    print("  mixed     - Cycles through all scenarios")
    print()

    run_simulation(scenario=args.scenario, duration=args.duration)


if __name__ == '__main__':
    main()
