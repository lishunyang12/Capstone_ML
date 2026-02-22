#ifndef SIAMESE_LSTM_H
#define SIAMESE_LSTM_H

#include "../hls_weights/model_params.h"

// Maximum sequence length for local buffer allocation
#define MAX_SEQ_LEN SEQ_LEN  // 450

// Combined feature dimension: emb1 + emb2 + |emb1-emb2|
#define COMBINED_DIM (EMBEDDING_DIM * 3)  // 48

// Top-level function — synthesized to IP core
void siamese_lstm_top(
    const float *seq1,   // m_axi bundle=gmem0, depth=MAX_SEQ_LEN*INPUT_DIM
    const float *seq2,   // m_axi bundle=gmem1, depth=MAX_SEQ_LEN*INPUT_DIM
    float *result,       // m_axi bundle=gmem2, depth=1
    int seq1_len,        // s_axilite (1..450)
    int seq2_len         // s_axilite (1..450)
);

#endif // SIAMESE_LSTM_H
