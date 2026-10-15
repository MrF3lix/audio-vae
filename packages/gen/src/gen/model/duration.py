"""Total-length predictor: text features -> log number of latent frames.

Without a token-level alignment only the utterance length is predicted; where in time each word
lands is up to the generator's cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gen.model.modules import TransformerBlock, masked_mean


class TotalDurationPredictor(nn.Module):
    def __init__(self, text_dim: int, hidden: int = 256, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden)
        self.blocks = nn.ModuleList([TransformerBlock(hidden, n_heads, 4, 3, dropout) for _ in range(n_layers)])
        self.attn_pool = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, text: Tensor, text_mask: Tensor) -> Tensor:
        """`text (B, N, D)`, `text_mask (B, N)` -> log latent frames `(B,)`."""
        x = self.proj(text) * text_mask[..., None]
        for block in self.blocks:
            x = block(x, text_mask)
        scores = self.attn_pool(x).squeeze(-1).masked_fill(~text_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)[..., None]
        log_n = torch.log(text_mask.sum(1, keepdim=True).float())  # text length is the strongest cue
        pooled = torch.cat([(attn * x).sum(1), masked_mean(x, text_mask), log_n], dim=-1)
        return self.head(pooled).squeeze(-1) + log_n.squeeze(-1)

    @staticmethod
    def loss(log_pred: Tensor, lengths: Tensor) -> Tensor:
        return F.mse_loss(log_pred, torch.log(lengths.float()))
