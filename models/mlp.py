import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """
    Drop-in replacement for AttentionLayer that uses a per-token MLP.
    Returns (output, None) so the rest of the model keeps working.
    """
    def __init__(self, d_model, hidden_mul=4, dropout=0.1, mix=False, **kwargs):
        """
        Parameters
        ----------
        d_model     : int   – token/feature size (same as in AttentionLayer)
        hidden_mul  : int   – expansion factor for the hidden layer (default 4× like Transformer FFN)
        dropout     : float – dropout rate between the two linear layers
        mix         : bool  – kept only so you can flip between AttentionLayer and MLPBlock without code changes
        **kwargs    : ignored – lets you pass n_heads etc. without breaking ctor
        """
        super().__init__()
        hidden = hidden_mul * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mix = mix           # no effect here, but preserves the interface

    def forward(self, queries, keys=None, values=None, attn_mask=None):
        """
        Everything except `queries` is ignored so the caller
        doesn't have to be rewritten.
        """
        x = queries                                # (B, L, d_model)
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)

        # Interface match: return attn-like object
        return x, None
