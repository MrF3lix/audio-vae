"""Transformer over latent frames, conditioned on text by cross-attention.

Shared by the generative objectives: flow matching uses it time-conditioned (adaLN-zero) with a
velocity head `(B, C, L)`; an energy model can use it time-independent (`time_conditioned=False`)
with a scalar head. Frames are tokens; positions enter through rotary self-attention and a conv
position embedding, the text only through cross-attention (no monotonic alignment: the Standard
German text and the Bernese speech differ in word order and vocabulary).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gen.model.modules import ConvPositionEmbedding, DiTBlock, FinalLayer, RotaryEmbedding, TimestepEmbedding


class LatentTransformer(nn.Module):
    def __init__(self, latent_dim: int, text_dim: int, out_dim: int | None = None, d_model: int = 384, depth: int = 8,
                 n_heads: int = 6, ff_mult: int = 4, dropout: float = 0.1, conv_pos_kernel: int = 31,
                 time_conditioned: bool = True):
        super().__init__()
        self.time_conditioned = time_conditioned
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.conv_pos = ConvPositionEmbedding(d_model, conv_pos_kernel)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.time_emb = TimestepEmbedding(d_model) if time_conditioned else None
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, ff_mult, dropout, adaln=time_conditioned) for _ in range(depth)])
        out_dim = latent_dim if out_dim is None else out_dim
        if time_conditioned:
            self.final = FinalLayer(d_model, out_dim)
        else:
            self.final = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, z: Tensor, mask: Tensor, text: Tensor, text_mask: Tensor, t: Tensor | None = None) -> Tensor:
        """`z (B, C, L)`, `mask (B, L)`, `text (B, N, D_text)`, `text_mask (B, N)`, `t (B,)` -> `(B, out_dim, L)`."""
        x = self.in_proj(z.transpose(1, 2)) * mask[..., None]
        x = x + self.conv_pos(x, mask)
        mem = self.text_proj(text)
        c = self.time_emb(t) if self.time_conditioned else None
        rope = self.rope(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, c, mask, mem, text_mask, rope)
        out = self.final(x, c) if self.time_conditioned else self.final(x)
        return (out * mask[..., None]).transpose(1, 2)
