#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

// Model architecture constants
#define INPUT_DIM       7    // ax, ay, az, gx, gy, gz, flex
#define HIDDEN_DIM     32    // LSTM hidden size
#define EMBEDDING_DIM  16    // Projection output size
#define SIM_FC1_OUT    16    // Similarity head FC1 output
#define SIM_FC2_OUT     1    // Final output (similarity score)
#define NUM_GATES       4    // LSTM gates: i, f, g, o

// Input constants
#define SEQ_LEN       450    // 3 seconds at 150Hz
#define MIN_SEQ_LEN   150    // 1 second minimum
#define SAMPLING_HZ   150    // Sensor sampling rate

// Derived constants
#define LSTM_GATE_SIZE (NUM_GATES * HIDDEN_DIM)  // 128
#define SIM_INPUT_DIM  (EMBEDDING_DIM * 3)       // 48 (emb1 + emb2 + |diff|)

#endif // MODEL_PARAMS_H