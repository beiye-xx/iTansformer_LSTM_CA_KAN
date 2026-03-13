import torch
import torch.nn as nn


class GTU(nn.Module):
    """Gated Tanh Unit (GTU) fusion module.

    Replaces cross-attention for fusing iTransformer and LSTM features.
    Uses element-wise gating: output = tanh(W_t * x1) * sigmoid(W_g * proj(x2))
    """

    def __init__(self, dim, length):
        """
        Args:
            dim:    feature dimension of the iTransformer output (query side)
            length: hidden size of the LSTM output (key/value side)
        """
        super(GTU, self).__init__()
        self.proj = nn.Linear(length, dim)
        self.tanh_gate = nn.Linear(dim, dim)
        self.sig_gate = nn.Linear(dim, dim)

    def forward(self, input1, input2):
        # input1: (B, C1, dim)   – iTransformer token features
        # input2: (B, T, length) – LSTM sequence output
        # Use the last LSTM hidden state and project to dim
        x2 = self.proj(input2[:, -1, :])          # (B, dim)
        x2 = x2.unsqueeze(1).expand_as(input1)    # (B, C1, dim)
        return torch.tanh(self.tanh_gate(input1)) * torch.sigmoid(self.sig_gate(x2))
