"""What every text-conditioned latent model shares: the text encoder, the total-length predictor,
and the learned null text for classifier-free guidance.

Subclasses add a backbone and implement `training_step`, `validation_losses` and `synthesise`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gen.model.duration import TotalDurationPredictor


class TextConditionedLatentModel(nn.Module):
    def __init__(self, text_encoder: nn.Module, latent_dim: int, duration: dict | None = None,
                 p_uncond: float = 0.1, duration_weight: float = 1.0):
        super().__init__()
        self.latent_dim, self.p_uncond, self.duration_weight = latent_dim, p_uncond, duration_weight
        self.text_encoder = text_encoder
        self.duration = TotalDurationPredictor(text_encoder.dim, **(duration or {}))
        self.null_text = nn.Parameter(torch.randn(1, 1, text_encoder.dim) * text_encoder.dim**-0.5)

    @property
    def device(self) -> torch.device:
        return self.null_text.device

    def encode_text(self, texts: list[str], device) -> tuple[Tensor, Tensor]:
        return self.text_encoder.encode(texts, device)

    def null_condition(self, h: Tensor, h_mask: Tensor) -> tuple[Tensor, Tensor]:
        """The unconditional input: one learned token, the rest masked."""
        null = self.null_text.to(h.dtype).expand(h.shape[0], h.shape[1], -1)
        null_mask = torch.zeros_like(h_mask)
        null_mask[:, 0] = True
        return null * null_mask[..., None], null_mask

    def drop_condition(self, h: Tensor, h_mask: Tensor, drop: Tensor) -> tuple[Tensor, Tensor]:
        null, null_mask = self.null_condition(h, h_mask)
        return torch.where(drop[:, None, None], null, h), torch.where(drop[:, None], null_mask, h_mask)

    def random_drop(self, h: Tensor, h_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.p_uncond <= 0:
            return h, h_mask
        drop = torch.rand(h.shape[0], device=h.device) < self.p_uncond
        return self.drop_condition(h, h_mask, drop)

    def duration_loss(self, h: Tensor, h_mask: Tensor, z_lengths: Tensor) -> Tensor:
        # The length predictor sees detached text features: its loss should not shape the encoder.
        return self.duration.loss(self.duration(h.detach(), h_mask), z_lengths)

    def predict_lengths(self, h: Tensor, h_mask: Tensor, length_scale: float = 1.0) -> Tensor:
        return torch.round(torch.exp(self.duration(h, h_mask)) * length_scale).clamp_min(1).long()

    def initial_noise(self, n: int, length: int, generator: torch.Generator | None, dtype) -> Tensor:
        shape = (n, self.latent_dim, length)
        z = torch.randn(shape, generator=generator, device="cpu" if generator is not None else self.device)
        return z.to(self.device, dtype)
