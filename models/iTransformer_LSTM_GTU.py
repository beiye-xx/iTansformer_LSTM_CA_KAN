from models import iTransformer_block
from models.gtu import GTU
import torch.nn as nn
from models.KAN import KAN


class iTransformer_LSTM_GTU(nn.Module):
    """iTransformer + LSTM fused via Gated Tanh Unit (GTU) + KAN head.

    Drop-in replacement for iTransformer_LSTM that swaps cross-attention for GTU.
    """

    def __init__(self, input_size=5, length_pre=1, dim_lstm=128, depth_lstm=3,
                 length_input=48, dim_embed=128, depth=4, heads=6):
        super(iTransformer_LSTM_GTU, self).__init__()
        self.model1 = iTransformer_block(
            num_variates=1, lookback_len=length_input,
            pred_length=length_pre, dim=dim_embed, depth=depth, heads=heads,
            num_tokens_per_variate=1, use_reversible_instance_norm=True)
        self.lstm = nn.LSTM(input_size=input_size - 1,
                            hidden_size=dim_lstm,
                            num_layers=depth_lstm,
                            batch_first=True,
                            bidirectional=False)
        self.gtu = GTU(dim=dim_embed, length=dim_lstm)
        self.k_mpl = KAN([dim_embed, length_pre])

    def forward(self, x):
        x2, _ = self.lstm(x[:, :, 1:])
        x1 = self.model1(x[:, :, 0, None])
        x1 = self.gtu(x1, x2)
        output = self.k_mpl(x1)
        return output[:, 0, :]
