# Tai Chi Motion Sync - AI Module

## Overview

Real-time motion similarity detection for a Tai Chi AR training system. Compares motion between a **Shifu** (instructor) and **Apprentice** (learner) using wearable IMU + flex sensors.

**Project:** CG4002 Computer Engineering Capstone
**Target:** Ultra96-V2 (Zynq UltraScale+ ZU3EG)

---

## End-to-End Pipeline

```
STEP 1          STEP 2            STEP 3           STEP 4          STEP 5
Train Model --> Export Weights --> HLS C Sim  ----> Vivado Build --> Deploy to FPGA
  (sw/)         (fpga/)            (fpga/)          (fpga/)         (fpga/deploy/)
```

If you retrain or change the model, re-run from Step 2 onwards.

---

## Project Structure

```
prototype/
├── README.md
├── trained_model.pth              # Trained model checkpoint
│
├── sw/                            # STEP 1: Software (PyTorch)
│   ├── model.py                   #   Siamese LSTM architecture
│   ├── dataset.py                 #   Data loading utilities
│   ├── train.py                   #   Training script
│   ├── evaluate.py                #   Evaluation vs baselines
│   ├── inference.py               #   Batch inference
│   ├── metrics.py                 #   Phase offset calculation
│   ├── online_serving.py          #   Real-time predictor
│   ├── simulate_online.py         #   Simulation demo
│   ├── visualize.py               #   Visualization generation
│   └── generate_synthetic.py      #   Synthetic data generator
│
├── data/                          #   Training/test data
│   ├── train/pairs.npz
│   ├── val/pairs.npz
│   └── test/pairs.npz
│
├── visualizations/                #   Generated plots
│
├── fpga/                          # STEPS 2-5: FPGA Pipeline
│   ├── export_weights.py          #   STEP 2: PyTorch -> C headers
│   ├── generate_test_vectors.py   #   STEP 2: PyTorch -> test vectors
│   ├── verify_hls_logic.py        #   STEP 2: NumPy verification
│   │
│   ├── hls/                       #   STEP 3: HLS C++ source
│   │   ├── siamese_lstm.h
│   │   ├── siamese_lstm.cpp
│   │   ├── siamese_lstm_tb.cpp
│   │   └── test_vectors/          #   Generated test data
│   │
│   ├── hls_weights/               #   Generated C weight headers
│   │   ├── model_params.h
│   │   ├── weights_lstm.h
│   │   ├── weights_projection.h
│   │   └── weights_similarity.h
│   │
│   ├── run_csim.tcl               #   STEP 3: Vitis HLS C simulation
│   ├── build_bitstream.tcl        #   STEP 4: Vivado bitstream build
│   │
│   └── deploy/                    #   STEP 5: FPGA deployment
│       ├── siamese_lstm.bit       #   Bitstream
│       ├── siamese_lstm.hwh       #   Hardware handoff
│       ├── test_siamese_lstm.py   #   On-board test script
│       └── upload_to_board.py     #   Upload script
│
└── (gitignored)
    siamese_lstm_hls/              # Vitis HLS project (generated)
    siamese_lstm_hls2/             # Vitis HLS project (generated)
    siamese_lstm_vivado/           # Vivado project (generated)
    siamese_lstm_vivado2/          # Vivado project (generated)
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

| Parameter | Value |
|-----------|-------|
| input_dim | 7 (ax, ay, az, gx, gy, gz, flex) |
| hidden_dim | 32 |
| embedding_dim | 16 |
| dropout | 0.2 |
| sampling_rate | 150 Hz |
| window | 450 samples (3 sec) |

---

## How to Run

### Prerequisites
```bash
pip install torch numpy scipy matplotlib
```

### Step 1: Train the Model (sw/)

```bash
# Generate synthetic data (if no real data)
python sw/generate_synthetic.py --output data --n-train 200 --n-val 50 --n-test 50

# Train
python sw/train.py --data-dir data --epochs 100 --batch-size 16
# Output: trained_model.pth

# Evaluate
python sw/evaluate.py --checkpoint trained_model.pth --data-dir data

# Visualize
python sw/visualize.py --checkpoint trained_model.pth --data-dir data
```

### Step 2: Export Weights for HLS (fpga/)

```bash
# Export model weights to C headers
python fpga/export_weights.py --checkpoint trained_model.pth --output-dir fpga/hls_weights

# Generate test vectors for C simulation
python fpga/generate_test_vectors.py --checkpoint trained_model.pth --output-dir fpga/hls/test_vectors

# Verify HLS logic matches PyTorch
python fpga/verify_hls_logic.py
```

### Step 3: HLS C Simulation (fpga/)

```bash
cd fpga
vitis_hls -f run_csim.tcl
```

### Step 4: Build Bitstream (fpga/)

In Vivado Tcl Console:
```tcl
source C:/Users/lsy/Downloads/Projects/Capstone/prototype/fpga/build_bitstream.tcl
```

### Step 5: Deploy to FPGA (fpga/deploy/)

```bash
python fpga/deploy/upload_to_board.py
```

Then SSH to the board and run:
```bash
ssh xilinx@makerslab-fpga-43.ddns.comp.nus.edu.sg
python3 test_siamese_lstm.py
```

---

## Output Format

| Field | Type | Description |
|-------|------|-------------|
| `similarity_score` | float [0, 1] | How similar the movements are |
| `phase_offset_ms` | float | Temporal offset in milliseconds |
| `feedback.sync` | string | excellent / good / needs_improvement / out_of_sync |
| `ar_color` | string | green / yellow / red |

### Thresholds
```
similarity >= 0.8  ->  excellent (green)
similarity >= 0.6  ->  good (green)
similarity >= 0.4  ->  needs_improvement (yellow)
similarity <  0.4  ->  out_of_sync (red)
```

---

## Online Serving API

```python
from sw.online_serving import OnlinePredictor

predictor = OnlinePredictor(
    checkpoint_path='trained_model.pth',
    window_size=450,
    min_samples=150
)

# Feed sensor data at 150Hz
predictor.add_shifu_reading(shifu_data)
predictor.add_apprentice_reading(apprentice_data)

# Get prediction at 10Hz
result = predictor.predict()
```

### Real-time Simulation
```bash
python sw/simulate_online.py --scenario similar --duration 30
python sw/simulate_online.py --scenario delayed --duration 30
python sw/simulate_online.py --scenario mixed --duration 60
```
