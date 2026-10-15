"""Text-conditioned flow matching in the VAE's (normalised) latent space.

    text ─ TextEncoder ─► h (N, D) ─┬─ TotalDurationPredictor ─► L (number of latent frames)
                                    └─ cross-attention ──┐
    z_0 ~ N(0, I) (B, C, L) ── ODE with v_θ(z_t, t, h) ──┴─► z_1 ─► VAE decoder ─► mel ─► vocoder

Losses: conditional flow matching on the OT path `z_t = (1 - (1 - σ) t) z_0 + t z_1` with
target `z_1 - (1 - σ) z_0`, `t` logit-normal (SD3), masked to valid frames; plus the MSE of
the log total length. The text is replaced by a learned null embedding with probability
`p_uncond`, which enables classifier-free guidance at sampling time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from common.lengths import sequence_mask
from gen.model.backbone import LatentTransformer
from gen.model.base import TextConditionedLatentModel


class LatentFlowMatching(TextConditionedLatentModel):
    def __init__(self, text_encoder: nn.Module, latent_dim: int = 32, backbone: dict | None = None,
                 duration: dict | None = None, sigma_min: float = 1e-4, p_uncond: float = 0.1,
                 t_sampling: str = "logit_normal", duration_weight: float = 1.0):
        super().__init__(text_encoder, latent_dim, duration, p_uncond, duration_weight)
        if t_sampling not in ("uniform", "logit_normal"):
            raise ValueError(f"t_sampling must be uniform or logit_normal, got {t_sampling!r}")
        self.sigma_min, self.t_sampling = sigma_min, t_sampling
        self.backbone = LatentTransformer(latent_dim, text_encoder.dim, **(backbone or {}))

    # ------------------------------------------------------------------ training
    def sample_t(self, n: int, device, dtype=torch.float32) -> Tensor:
        if self.t_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device, dtype=dtype))
        return torch.rand(n, device=device, dtype=dtype)

    def flow_loss(self, z1: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor,
                  noise: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        m = mask[:, None, :].to(z1.dtype)
        z0 = torch.randn_like(z1) if noise is None else noise
        t = self.sample_t(z1.shape[0], z1.device, z1.dtype) if t is None else t
        tt = t[:, None, None]
        zt = (1 - (1 - self.sigma_min) * tt) * z0 + tt * z1
        target = z1 - (1 - self.sigma_min) * z0
        v = self.backbone(zt * m, mask, h, h_mask, t)
        return ((v - target).square() * m).sum() / (m.sum() * self.latent_dim)

    def training_step(self, batch: dict, noise: Tensor | None = None, t: Tensor | None = None,
                      cond_drop: bool = True) -> dict:
        z, z_lengths = batch["z"], batch["z_lengths"]
        mask = sequence_mask(z_lengths, z.shape[-1])
        h, h_mask = self.encode_text(batch["texts"], z.device)
        dur_loss = self.duration_loss(h, h_mask, z_lengths)
        if cond_drop:
            h, h_mask = self.random_drop(h, h_mask)
        fm_loss = self.flow_loss(z, mask, h, h_mask, noise, t)
        return {"loss": fm_loss + self.duration_weight * dur_loss, "fm": fm_loss, "dur": dur_loss}

    def validation_losses(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor, noise: Tensor, t: Tensor) -> dict:
        """Per-frame validation metrics at fixed noise and times (comparable across runs)."""
        return {"fm": self.flow_loss(z, mask, h, h_mask, noise, t)}

    # ------------------------------------------------------------------ inference

    def velocity(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor, t: Tensor,
                 guidance_scale: float = 1.0) -> Tensor:
        if guidance_scale == 1.0:
            return self.backbone(z, mask, h, h_mask, t)
        null, null_mask = self.null_condition(h, h_mask)
        v = self.backbone(torch.cat([z, z]), torch.cat([mask, mask]), torch.cat([h, null]),
                          torch.cat([h_mask, null_mask]), torch.cat([t, t]))
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def synthesise(self, texts: list[str], n_steps: int = 32, guidance_scale: float = 2.0, temperature: float = 1.0,
                   length_scale: float = 1.0, z_lengths: Tensor | None = None, method: str = "midpoint",
                   generator: torch.Generator | None = None) -> dict:
        """Texts -> normalised latents `z (B, C, L)` and `z_lengths (B,)`.

        `z_lengths` overrides the length predictor (e.g. the ground-truth lengths, to judge the
        generator independently of it). `method`: `euler` or `midpoint` (2 evaluations per step).
        """
        device = self.device
        h, h_mask = self.encode_text(texts, device)
        if z_lengths is None:
            z_lengths = self.predict_lengths(h, h_mask, length_scale)
        z_lengths = z_lengths.to(device)
        mask = sequence_mask(z_lengths, int(z_lengths.max()))
        m = mask[:, None, :].to(h.dtype)

        z = self.initial_noise(len(texts), mask.shape[1], generator, h.dtype) * temperature * m

        def v(z_, t_):
            return self.velocity(z_, mask, h, h_mask, torch.full((z_.shape[0],), t_, device=device, dtype=h.dtype),
                                 guidance_scale) * m

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = i * dt
            if method == "euler":
                z = z + dt * v(z, t)
            elif method == "midpoint":
                z = z + dt * v(z + 0.5 * dt * v(z, t), t + 0.5 * dt)
            else:
                raise ValueError(f"method must be euler or midpoint, got {method!r}")
        return {"z": z * m, "z_lengths": z_lengths}
