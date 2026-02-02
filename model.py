"""
Siamese LSTM Model for Tai Chi Motion Similarity

Input: Two sequences of shape [seq_len, 7]
       Features: ax, ay, az, gx, gy, gz, flex
Output: Similarity score [0, 1]
"""

import torch
import torch.nn as nn


class SiameseLSTM(nn.Module):
    """
    Siamese LSTM for comparing two motion sequences.

    Architecture:
        - Shared LSTM encoder for both sequences
        - Projection to embedding space
        - Similarity head computes final score
    """

    def __init__(
        self,
        input_dim: int = 7,      # ax, ay, az, gx, gy, gz, flex
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.projection = nn.Linear(hidden_dim, embedding_dim)

        self.similarity_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence to embedding."""
        _, (h_n, _) = self.lstm(x)
        return self.projection(h_n[-1])

    def forward(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seq1: [batch, seq_len, 7] - control/shifu sequence
            seq2: [batch, seq_len, 7] - test/apprentice sequence
        Returns:
            similarity: [batch, 1] in [0, 1]
        """
        emb1 = self.encode(seq1)
        emb2 = self.encode(seq2)

        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([emb1, emb2, diff], dim=-1)

        return self.similarity_head(combined)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = SiameseLSTM()
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    seq1 = torch.randn(2, 450, 7)  # batch=2, seq_len=450, features=7
    seq2 = torch.randn(2, 450, 7)
    out = model(seq1, seq2)
    print(f"Input shape: {seq1.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out.flatten().tolist()}")
