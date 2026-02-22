/**
 * Siamese LSTM — Vitis HLS Implementation
 *
 * Compares two IMU motion sequences and produces a similarity score [0,1].
 * Target: Ultra96-V2 (Zynq UltraScale+ ZU3EG)
 *
 * Architecture:
 *   1. LSTM encoder (shared weights) produces 16-d embedding per sequence
 *   2. Similarity head: concat [emb1, emb2, |emb1-emb2|] → FC1(48→16)+ReLU → FC2(16→1)+Sigmoid
 */

#include "siamese_lstm.h"
#include "../hls_weights/weights_lstm.h"
#include "../hls_weights/weights_projection.h"
#include "../hls_weights/weights_similarity.h"
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Sigmoid & tanh helpers (use float math — HLS synthesizes to LUT/DSP)
// ---------------------------------------------------------------------------
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// LSTM encoder: processes a sequence and writes a 16-d embedding
//
// For each timestep:
//   gates[128] = W_ih * x_t + W_hh * h + bias   (combined bias)
//   i = sigmoid(gates[0..31])
//   f = sigmoid(gates[32..63])
//   g = tanh(gates[64..95])
//   o = sigmoid(gates[96..127])
//   c = f*c + i*g
//   h = o * tanh(c)
//
// After all timesteps:
//   embedding = proj_weight * h + proj_bias      (Linear 32→16)
// ---------------------------------------------------------------------------
static void lstm_encode(
    const float *seq,     // [seq_len * INPUT_DIM], flat row-major in local BRAM
    int seq_len,
    float embedding[EMBEDDING_DIM]
) {
    // LSTM state
    float h[HIDDEN_DIM];
    float c[HIDDEN_DIM];

    // Zero-init hidden state
    INIT_STATE:
    for (int i = 0; i < HIDDEN_DIM; i++) {
#pragma HLS PIPELINE II=1
        h[i] = 0.0f;
        c[i] = 0.0f;
    }

    // Process each timestep
    TIMESTEP:
    for (int t = 0; t < seq_len; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=450

        // Load current input vector (7 features)
        float x_t[INPUT_DIM];
#pragma HLS ARRAY_PARTITION variable=x_t complete
        for (int i = 0; i < INPUT_DIM; i++) {
#pragma HLS UNROLL
            x_t[i] = seq[t * INPUT_DIM + i];
        }

        // Compute all 128 gate pre-activations: gates = W_ih * x_t + W_hh * h + bias
        float gates[LSTM_GATE_SIZE];

        GATE_COMPUTE:
        for (int g = 0; g < LSTM_GATE_SIZE; g++) {
            // No PIPELINE here — let inner loops share multipliers to save DSP
            float acc = lstm_bias[g];

            // W_ih[g] dot x_t
            GATE_IH:
            for (int j = 0; j < INPUT_DIM; j++) {
#pragma HLS PIPELINE II=1
                acc += lstm_weight_ih[g * INPUT_DIM + j] * x_t[j];
            }

            // W_hh[g] dot h
            GATE_HH:
            for (int j = 0; j < HIDDEN_DIM; j++) {
#pragma HLS PIPELINE II=1
                acc += lstm_weight_hh[g * HIDDEN_DIM + j] * h[j];
            }

            gates[g] = acc;
        }

        // Apply activations and update state
        // PyTorch gate order: i[0..31], f[32..63], g[64..95], o[96..127]
        UPDATE_STATE:
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float ig = sigmoid(gates[i]);                       // input gate
            float fg = sigmoid(gates[HIDDEN_DIM + i]);          // forget gate
            float gg = tanhf(gates[2 * HIDDEN_DIM + i]);        // cell gate
            float og = sigmoid(gates[3 * HIDDEN_DIM + i]);      // output gate

            c[i] = fg * c[i] + ig * gg;
            h[i] = og * tanhf(c[i]);
        }
    }

    // Projection: embedding = proj_weight * h + proj_bias  (Linear [16,32] * [32] + [16])
    PROJECTION:
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float acc = proj_bias[i];
        PROJ_DOT:
        for (int j = 0; j < HIDDEN_DIM; j++) {
#pragma HLS PIPELINE II=1
            acc += proj_weight[i * HIDDEN_DIM + j] * h[j];
        }
        embedding[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Similarity head: [emb1, emb2, |emb1-emb2|] → FC1(48→16)+ReLU → FC2(16→1)+Sigmoid
// ---------------------------------------------------------------------------
static float similarity_head(
    const float emb1[EMBEDDING_DIM],
    const float emb2[EMBEDDING_DIM]
) {
    // Build combined vector [emb1, emb2, |emb1-emb2|]
    float combined[COMBINED_DIM];

    BUILD_COMBINED:
    for (int i = 0; i < EMBEDDING_DIM; i++) {
#pragma HLS PIPELINE II=1
        combined[i] = emb1[i];
        combined[EMBEDDING_DIM + i] = emb2[i];
        float diff = emb1[i] - emb2[i];
        combined[2 * EMBEDDING_DIM + i] = (diff >= 0.0f) ? diff : -diff;
    }

    // FC1: Linear(48, 16) + ReLU
    float fc1_out[SIM_FC1_OUT];

    FC1:
    for (int i = 0; i < SIM_FC1_OUT; i++) {
        float acc = sim_fc1_bias[i];
        FC1_DOT:
        for (int j = 0; j < COMBINED_DIM; j++) {
#pragma HLS PIPELINE II=1
            acc += sim_fc1_weight[i * COMBINED_DIM + j] * combined[j];
        }
        fc1_out[i] = (acc > 0.0f) ? acc : 0.0f;  // ReLU
    }

    // FC2: Linear(16, 1) + Sigmoid
    float acc = sim_fc2_bias[0];
    FC2:
    for (int j = 0; j < SIM_FC1_OUT; j++) {
#pragma HLS PIPELINE II=1
        acc += sim_fc2_weight[j] * fc1_out[j];
    }

    return sigmoid(acc);
}

// ---------------------------------------------------------------------------
// Top-level function — AXI interface for PYNQ DMA
// ---------------------------------------------------------------------------
void siamese_lstm_top(
    const float *seq1,
    const float *seq2,
    float *result,
    int seq1_len,
    int seq2_len
) {
    // AXI interface pragmas
#pragma HLS INTERFACE m_axi port=seq1   bundle=gmem0 depth=3150
#pragma HLS INTERFACE m_axi port=seq2   bundle=gmem1 depth=3150
#pragma HLS INTERFACE m_axi port=result bundle=gmem2 depth=1
#pragma HLS INTERFACE s_axilite port=seq1_len
#pragma HLS INTERFACE s_axilite port=seq2_len
#pragma HLS INTERFACE s_axilite port=return

    // Clamp lengths to valid range
    if (seq1_len < 1) seq1_len = 1;
    if (seq1_len > MAX_SEQ_LEN) seq1_len = MAX_SEQ_LEN;
    if (seq2_len < 1) seq2_len = 1;
    if (seq2_len > MAX_SEQ_LEN) seq2_len = MAX_SEQ_LEN;

    // Local BRAM buffer for one sequence at a time (reused)
    float local_seq[MAX_SEQ_LEN * INPUT_DIM];

    float emb1[EMBEDDING_DIM];
    float emb2[EMBEDDING_DIM];

    // --- Encode sequence 1 ---
    // Burst-read seq1 from DDR to local BRAM
    memcpy(local_seq, seq1, seq1_len * INPUT_DIM * sizeof(float));
    lstm_encode(local_seq, seq1_len, emb1);

    // --- Encode sequence 2 ---
    // Burst-read seq2 from DDR to local BRAM (reuse buffer)
    memcpy(local_seq, seq2, seq2_len * INPUT_DIM * sizeof(float));
    lstm_encode(local_seq, seq2_len, emb2);

    // --- Compute similarity ---
    float score = similarity_head(emb1, emb2);

    // Write result to DDR
    result[0] = score;
}
