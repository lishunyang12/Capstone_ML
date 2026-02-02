"""
Online Serving Module for Real-Time Inference

Architecture:
    Shifu Sensors  ──┐
                     ├──> Buffer ──> Sliding Window ──> Model ──> Feedback ──> AR Visualizer
    Apprentice Sensors ┘

Features:
    - Sliding window inference
    - Configurable window size and stride
    - Low-latency prediction
    - Thread-safe data buffering
"""

import torch
import numpy as np
import time
import threading
from collections import deque
from typing import Dict, Optional, Callable
from dataclasses import dataclass

from model import SiameseLSTM
from metrics import compute_all_metrics


@dataclass
class SensorReading:
    """Single sensor reading from IMU + Flex."""
    timestamp_ms: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    flex: float

    def to_array(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az, self.gx, self.gy, self.gz, self.flex], dtype=np.float32)


class SensorBuffer:
    """Thread-safe circular buffer for sensor data."""

    def __init__(self, max_size: int = 450):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, reading: np.ndarray):
        """Add a sensor reading [7] to buffer."""
        with self.lock:
            self.buffer.append(reading)

    def get_window(self, size: int) -> Optional[np.ndarray]:
        """Get the last `size` readings as array [size, 7]."""
        with self.lock:
            if len(self.buffer) < size:
                return None
            data = list(self.buffer)[-size:]
            return np.array(data, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()


class OnlinePredictor:
    """
    Real-time motion similarity predictor.

    Usage:
        predictor = OnlinePredictor('trained_model.pth')

        # In your data receiving loop:
        predictor.add_shifu_reading(sensor_data)
        predictor.add_apprentice_reading(sensor_data)

        # Get prediction when ready:
        result = predictor.predict()
    """

    def __init__(
        self,
        checkpoint_path: str,
        window_size: int = 450,      # 3 seconds at 150Hz
        min_samples: int = 150,      # Minimum samples before prediction (1 second)
        device: str = None
    ):
        """
        Args:
            checkpoint_path: Path to trained model
            window_size: Number of samples for inference window
            min_samples: Minimum samples needed before first prediction
            device: 'cuda' or 'cpu'
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.window_size = window_size
        self.min_samples = min_samples

        # Load model
        self.model = SiameseLSTM()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Buffers for shifu and apprentice
        self.shifu_buffer = SensorBuffer(max_size=window_size)
        self.apprentice_buffer = SensorBuffer(max_size=window_size)

        # Last prediction result
        self.last_result = None
        self.last_prediction_time = 0

        print(f"OnlinePredictor initialized")
        print(f"  Window size: {window_size} samples ({window_size/150:.1f}s)")
        print(f"  Min samples: {min_samples}")
        print(f"  Device: {device}")

    def add_shifu_reading(self, data: np.ndarray):
        """
        Add shifu sensor reading.
        Args:
            data: [7] array (ax, ay, az, gx, gy, gz, flex)
        """
        self.shifu_buffer.add(data)

    def add_apprentice_reading(self, data: np.ndarray):
        """
        Add apprentice sensor reading.
        Args:
            data: [7] array (ax, ay, az, gx, gy, gz, flex)
        """
        self.apprentice_buffer.add(data)

    def is_ready(self) -> bool:
        """Check if enough data for prediction."""
        return (len(self.shifu_buffer) >= self.min_samples and
                len(self.apprentice_buffer) >= self.min_samples)

    def predict(self) -> Optional[Dict]:
        """
        Run prediction on current buffer data.

        Returns:
            dict with similarity_score, phase_offset_ms, movement_quality, feedback
            or None if not enough data
        """
        if not self.is_ready():
            return None

        # Get current window size (use minimum of both buffers)
        current_size = min(len(self.shifu_buffer), len(self.apprentice_buffer))
        current_size = min(current_size, self.window_size)

        # Get windows
        shifu_window = self.shifu_buffer.get_window(current_size)
        apprentice_window = self.apprentice_buffer.get_window(current_size)

        if shifu_window is None or apprentice_window is None:
            return None

        # Pad to window_size if needed
        if current_size < self.window_size:
            shifu_window = self._pad_sequence(shifu_window, self.window_size)
            apprentice_window = self._pad_sequence(apprentice_window, self.window_size)

        # Run ML inference
        start_time = time.time()

        shifu_tensor = torch.from_numpy(shifu_window).float().unsqueeze(0).to(self.device)
        apprentice_tensor = torch.from_numpy(apprentice_window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            similarity_score = self.model(shifu_tensor, apprentice_tensor).item()

        inference_time_ms = (time.time() - start_time) * 1000

        # Compute additional metrics (hardcoded)
        result = compute_all_metrics(
            control_seq=shifu_window,
            test_seq=apprentice_window,
            similarity_score=similarity_score
        )

        # Add metadata
        result['inference_time_ms'] = round(inference_time_ms, 2)
        result['buffer_size'] = current_size
        result['timestamp'] = time.time()

        self.last_result = result
        self.last_prediction_time = time.time()

        return result

    def _pad_sequence(self, seq: np.ndarray, target_size: int) -> np.ndarray:
        """Pad sequence to target size by repeating first value."""
        current_size = seq.shape[0]
        if current_size >= target_size:
            return seq

        pad_size = target_size - current_size
        padding = np.tile(seq[0], (pad_size, 1))
        return np.vstack([padding, seq])

    def reset(self):
        """Clear buffers."""
        self.shifu_buffer.clear()
        self.apprentice_buffer.clear()
        self.last_result = None


class RealtimeServer:
    """
    Real-time serving with callback support.

    Runs prediction at fixed intervals and calls callback with results.
    """

    def __init__(
        self,
        predictor: OnlinePredictor,
        prediction_interval_ms: float = 100,  # Predict every 100ms
        callback: Callable[[Dict], None] = None
    ):
        self.predictor = predictor
        self.prediction_interval = prediction_interval_ms / 1000.0
        self.callback = callback or self._default_callback

        self.running = False
        self.prediction_thread = None

    def _default_callback(self, result: Dict):
        """Default callback - print result."""
        print(f"[{result['timestamp']:.2f}] "
              f"Similarity: {result['similarity_score']:.3f} | "
              f"Phase: {result['phase_offset_ms']:+.0f}ms | "
              f"Feedback: {result['feedback']['sync']}")

    def _prediction_loop(self):
        """Background prediction loop."""
        while self.running:
            result = self.predictor.predict()
            if result:
                self.callback(result)
            time.sleep(self.prediction_interval)

    def start(self):
        """Start prediction loop in background."""
        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()
        print(f"RealtimeServer started (interval: {self.prediction_interval*1000:.0f}ms)")

    def stop(self):
        """Stop prediction loop."""
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
        print("RealtimeServer stopped")


# =============================================================================
# Example: Simulated Real-time Serving
# =============================================================================

def simulate_realtime_serving():
    """Simulate real-time data streaming and prediction."""

    print("=" * 60)
    print("Real-time Serving Simulation")
    print("=" * 60)

    # Initialize predictor
    predictor = OnlinePredictor(
        checkpoint_path='trained_model.pth',
        window_size=450,
        min_samples=150
    )

    # Results storage for visualization
    results_log = []

    def on_prediction(result):
        results_log.append(result)
        color = '\033[92m' if result['ar_color'] == 'green' else '\033[91m'
        reset = '\033[0m'
        print(f"{color}[Score: {result['similarity_score']:.3f}] "
              f"{result['feedback']['sync']:<15} | "
              f"Phase: {result['phase_offset_ms']:+6.0f}ms{reset}")

    # Start server
    server = RealtimeServer(
        predictor=predictor,
        prediction_interval_ms=200,  # Predict every 200ms
        callback=on_prediction
    )

    # Generate simulated sensor data
    print("\nSimulating sensor data stream...")
    print("(Ctrl+C to stop)\n")

    # Load test data for simulation
    from dataset import MotionPairDataset
    dataset = MotionPairDataset('data', 'test')

    server.start()

    try:
        sample_idx = 0
        while True:
            # Get a sample
            shifu_seq = dataset.control_seqs[sample_idx % len(dataset)]
            apprentice_seq = dataset.test_seqs[sample_idx % len(dataset)]
            label = dataset.labels[sample_idx % len(dataset)]

            print(f"\n--- Streaming sample {sample_idx} (label: {label:.2f}) ---")

            # Simulate streaming at 150Hz
            for i in range(shifu_seq.shape[0]):
                predictor.add_shifu_reading(shifu_seq[i])
                predictor.add_apprentice_reading(apprentice_seq[i])
                time.sleep(1/150)  # 150Hz

            # Brief pause between samples
            predictor.reset()
            sample_idx += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        server.stop()

    return results_log


# =============================================================================
# Integration Example with MQTT (placeholder)
# =============================================================================

def mqtt_integration_example():
    """
    Example showing how to integrate with MQTT for real sensor data.

    This is a template - actual implementation depends on your MQTT setup.
    """
    code = '''
# Integration with MQTT (example)

import paho.mqtt.client as mqtt
from online_serving import OnlinePredictor, RealtimeServer

# Initialize predictor
predictor = OnlinePredictor('trained_model.pth')

# Callback for predictions
def on_prediction(result):
    # Send to AR visualizer via WebSocket
    ws_client.send(json.dumps(result))

# Start prediction server
server = RealtimeServer(predictor, prediction_interval_ms=100, callback=on_prediction)
server.start()

# MQTT callbacks
def on_shifu_data(client, userdata, msg):
    data = json.loads(msg.payload)
    reading = np.array([
        data['ax'], data['ay'], data['az'],
        data['gx'], data['gy'], data['gz'],
        data['flex']
    ], dtype=np.float32)
    predictor.add_shifu_reading(reading)

def on_apprentice_data(client, userdata, msg):
    data = json.loads(msg.payload)
    reading = np.array([
        data['ax'], data['ay'], data['az'],
        data['gx'], data['gy'], data['gz'],
        data['flex']
    ], dtype=np.float32)
    predictor.add_apprentice_reading(reading)

# Setup MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883)
mqtt_client.subscribe("sensors/shifu/#")
mqtt_client.subscribe("sensors/apprentice/#")
mqtt_client.loop_forever()
'''
    print(code)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--mqtt-example':
        mqtt_integration_example()
    else:
        simulate_realtime_serving()
