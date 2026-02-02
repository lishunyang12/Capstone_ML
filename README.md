# Tai Chi Motion Sync - AI Module

## Overview

This module provides real-time motion similarity detection for a Tai Chi AR training system. It compares motion between a **Shifu** (instructor) and **Apprentice** (learner) using wearable sensors.

**Project:** CG4002 Computer Engineering Capstone
**Status:** MVP/POC for midterm demo

---

## System Architecture

```
Shifu Sensors ────┐
                  ├──> Buffer ──> Sliding Window ──> Model ──> Feedback ──> AR Visualizer
Apprentice Sensors┘
```

### Input (Per Limb)
- 1 IMU sensor (accelerometer + gyroscope)
- 1 Flex sensor

### Data Format
Each sensor reading is a **7-element array**:
```
[ax, ay, az, gx, gy, gz, flex]
```

| Index | Field | Description | Unit |
|-------|-------|-------------|------|
| 0 | ax | Accelerometer X | g |
| 1 | ay | Accelerometer Y | g |
| 2 | az | Accelerometer Z | g |
| 3 | gx | Gyroscope X | deg/s |
| 4 | gy | Gyroscope Y | deg/s |
| 5 | gz | Gyroscope Z | deg/s |
| 6 | flex | Flex sensor | degrees |

### Sampling
- **Rate:** 150 Hz
- **Window:** 3 seconds (450 samples)
- **Minimum for prediction:** 1 second (150 samples)

---

## Output

### ML Output (Neural Network)
| Field | Type | Description |
|-------|------|-------------|
| `similarity_score` | float [0, 1] | How similar the movements are |

### Hardcoded Output (Cross-Correlation)
| Field | Type | Description |
|-------|------|-------------|
| `phase_offset_ms` | float | Temporal offset in milliseconds |

### Feedback (Derived)
| Field | Values | Description |
|-------|--------|-------------|
| `feedback.sync` | excellent / good / needs_improvement / out_of_sync | Based on similarity_score |
| `feedback.timing` | on_time / lagging / rushing | Based on phase_offset_ms |
| `ar_color` | green / yellow / red | For AR visualization |

### Thresholds
```
similarity >= 0.8  →  excellent (green)
similarity >= 0.6  →  good (green)
similarity >= 0.4  →  needs_improvement (yellow)
similarity <  0.4  →  out_of_sync (red)

|phase_offset| < 100ms  →  on_time
phase_offset < 0        →  lagging (apprentice behind)
phase_offset > 0        →  rushing (apprentice ahead)
```

---

## File Structure

```
prototype/
├── model.py              # Siamese LSTM neural network
├── dataset.py            # Data loading utilities
├── train.py              # Training script
├── evaluate.py           # Compare ML vs hardcoded baselines
├── inference.py          # Batch inference on test data
├── metrics.py            # Phase offset calculation
├── online_serving.py     # Real-time prediction server
├── simulate_online.py    # Interactive simulation demo
├── visualize.py          # Generate visualization plots
├── generate_synthetic.py # Create synthetic test data
├── trained_model.pth     # Trained model checkpoint
└── data/
    ├── train/pairs.npz   # Training data
    ├── val/pairs.npz     # Validation data
    └── test/pairs.npz    # Test data
```

---

## Model Architecture

**Siamese LSTM** - Two identical LSTM branches that encode Shifu and Apprentice sequences, then compare embeddings.

```
Shifu Seq [450, 7] ──> LSTM ──> Embedding [16]──┐
                                                ├──> Concat [48] ──> MLP ──> Similarity [0,1]
Apprentice Seq [450, 7] ──> LSTM ──> Embedding [16]──┘
                                                │
                                         |diff| [16]
```

### Model Parameters
```python
input_dim = 7        # Sensor features
hidden_dim = 32      # LSTM hidden size
embedding_dim = 16   # Output embedding size
dropout = 0.2
```

### Why Siamese LSTM?
- Handles variable-length sequences
- Learns temporal patterns
- Symmetric comparison (order doesn't matter)
- Small model suitable for edge deployment

---

## Data Format

### NPZ File Structure
```python
# data/{split}/pairs.npz
{
    'control_seqs': np.ndarray,  # Shape: [N, 450, 7] - Shifu sequences
    'test_seqs': np.ndarray,     # Shape: [N, 450, 7] - Apprentice sequences
    'labels': np.ndarray         # Shape: [N] - Similarity labels [0, 1]
}
```

### Label Convention
```
1.0 = Identical/very similar movement
0.8 = Similar with minor differences
0.4 = Different speed/timing
0.0 = Completely different movement
```

---

## How to Run

### Prerequisites
```bash
pip install torch numpy scipy matplotlib
```

### Step 1: Generate Synthetic Data (for testing pipeline)
```bash
python generate_synthetic.py --output data --n-train 200 --n-val 50 --n-test 50
```

### Step 2: Train the Model
```bash
python train.py --data-dir data --epochs 100 --batch-size 16
```
Output: `trained_model.pth`

### Step 3: Evaluate Model
```bash
python evaluate.py --checkpoint trained_model.pth --data-dir data
```
Compares ML model against hardcoded baselines (DTW, cosine similarity, etc.)

### Step 4: Run Inference
```bash
python inference.py --checkpoint trained_model.pth --data-dir data
```

### Step 5: Generate Visualizations
```bash
python visualize.py --checkpoint trained_model.pth --data-dir data
```
Output: `visualizations/` folder with plots

### Step 6: Run Real-time Simulation
```bash
# Test different scenarios
python simulate_online.py --scenario similar --duration 30
python simulate_online.py --scenario delayed --duration 30
python simulate_online.py --scenario different --duration 30
python simulate_online.py --scenario mixed --duration 60
```

| Scenario | Description | Expected Result |
|----------|-------------|-----------------|
| similar | Apprentice follows closely | Green, on_time |
| delayed | Apprentice 200ms behind | Green, lagging |
| different | Wrong movement | Red |
| fast | 1.5x speed | Yellow/Red |
| slow | 0.6x speed | Yellow/Red |
| mixed | Cycles through all | Varies |

Press `Ctrl+C` to stop.

---

## Online Serving API

### Usage
```python
from online_serving import OnlinePredictor

# Initialize
predictor = OnlinePredictor(
    checkpoint_path='trained_model.pth',
    window_size=450,      # 3 seconds at 150Hz
    min_samples=150       # 1 second minimum
)

# Feed sensor data (call at 150Hz)
predictor.add_shifu_reading(shifu_data)        # np.ndarray [7]
predictor.add_apprentice_reading(apprentice_data)  # np.ndarray [7]

# Get prediction (call at 10Hz)
result = predictor.predict()
if result:
    print(result['similarity_score'])   # 0.0 - 1.0
    print(result['phase_offset_ms'])    # milliseconds
    print(result['feedback']['sync'])   # excellent/good/needs_improvement/out_of_sync
    print(result['ar_color'])           # green/yellow/red
```

### With Callback Server
```python
from online_serving import OnlinePredictor, RealtimeServer

predictor = OnlinePredictor('trained_model.pth')

def on_result(result):
    # Send to AR visualizer
    send_to_ar(result['ar_color'], result['similarity_score'])

server = RealtimeServer(
    predictor=predictor,
    prediction_interval_ms=100,  # 10Hz predictions
    callback=on_result
)
server.start()

# ... feed sensor data ...

server.stop()
```

---

## Phase Offset Calculation

The timing/phase offset uses **cross-correlation** on gyroscope X signal:

1. Extract `gx` from both Shifu and Apprentice
2. Normalize signals (zero mean, unit variance)
3. Compute cross-correlation
4. Find lag at maximum correlation
5. Convert to milliseconds

```python
from metrics import compute_phase_offset

offset_ms = compute_phase_offset(shifu_seq, apprentice_seq, sampling_hz=150.0)
# Negative = apprentice lagging (behind)
# Positive = apprentice rushing (ahead)
```

---

## Hardcoded Baselines (for comparison)

| Method | Description | Performance |
|--------|-------------|-------------|
| angle_diff | Direct angle difference | Poor |
| cross_correlation | Signal correlation | Moderate |
| dtw | Dynamic Time Warping | Good |
| cosine_features | Cosine similarity on features | Moderate |
| **ML (Siamese LSTM)** | Learned similarity | **Best** |

ML shows ~95% improvement over best baseline (DTW) on synthetic data.

---

## Next Steps

1. **Collect Real Data** - Replace synthetic data with actual sensor recordings
2. **Multi-limb Support** - Extend to multiple IMU + flex sensors
3. **Model Optimization** - Quantization for edge deployment
4. **Integration** - Connect to MQTT/WebSocket for real sensor streams

---

## Troubleshooting

### Model not loading
```bash
# Check if trained_model.pth exists
ls trained_model.pth

# Retrain if missing
python train.py --data-dir data --epochs 100
```

### No prediction output
- Ensure at least 150 samples (1 second) in both buffers
- Check `predictor.is_ready()` returns True

### Simulation crashes
- Press Ctrl+C to gracefully stop
- Check ANSI terminal support on Windows

---

## Contact

For questions about the AI module, refer to this document or check the code comments.
