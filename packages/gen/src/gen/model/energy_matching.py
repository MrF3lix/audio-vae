"""Text-conditioned energy matching in the VAE's (normalised) latent space.

A time-independent scalar potential `V_θ(z; text)` whose negative gradient is the generating
field: `dz/dτ = -∇V(z)` carries noise to data (Balcerak et al., "Energy Matching"; the recipe of
`generative-modeling/src/gm/playground/11_mnist_energy_matching.ipynb`).

    V(z; h) = output_scale * Σ_valid frames e_θ(z, h)_l       (LatentTransformer, no time input)

Phase 1 (OT): minibatch-OT couple noise z_0 with data z_1, `z_t = (1 - t) z_0 + t z_1`,
`t ~ U[0, τ*]`, and regress `∇V(z_t)` onto `z_0 - z_1`. This needs the gradient of a gradient,
so attention runs on PyTorch's math kernel (the fused kernels have no double backward) and all
activations on the z path are smooth.

Phase 2 (from `cd_start_step` on, if `cd_weight > 0`): add contrastive divergence. Negatives are
Langevin chains under the same text, half started at the data (constant noise ε_max), half at
noise (ε schedule), for the first `neg_batch` utterances of the batch only (cost). Loss
`(V̄(data) - trimmed_mean V̄(neg)) / ε_max`, clamped at `-clamp_beta`, with V̄ the energy per
frame so utterances of different length are comparable. This shapes the data into minima of V, which makes the Langevin tail of the sampler (τ > τ*) meaningful.

Sampling: Langevin `z ← z - dτ ∇V + sqrt(2 dτ ε(τ)) ξ` for τ in [0, τ_s], ε = 0 until τ*, then
ramping to ε_max. With `tau_s = tau_star` and `eps_max = 0` it is the plain ODE.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.lengths import sequence_mask
from gen.model.backbone import LatentTransformer
from gen.model.base import TextConditionedLatentModel


def eps_schedule(tau: float, eps_max: float, tau_star: float) -> float:
    """No noise before τ*, linear ramp to ε_max over [τ*, 1] (if τ* < 1), ε_max after."""
    if tau < tau_star:
        return 0.0
    if tau_star < 1.0 and tau < 1.0:
        return eps_max * (tau - tau_star) / (1.0 - tau_star)
    return eps_max


def trimmed_mean(v: Tensor, alpha: float) -> Tensor:
    """Mean without the `alpha` fraction of largest values (outlier negatives)."""
    k = int(alpha * v.numel())
    return v.mean() if k == 0 else torch.sort(v)[0][: v.numel() - k].mean()


@torch.no_grad()
def ot_couple(z0: Tensor, z1: Tensor, mask: Tensor) -> Tensor:
    """Permute the noise `z0` to minimise Σ ||(z0_π(j) - z1_j) ⊙ mask_j||² over the batch.

    Noise is i.i.d. and padded like the data, so any noise sample can go to any utterance; the
    cost is measured on the target utterance's valid frames. Returns the permuted noise."""
    from scipy.optimize import linear_sum_assignment  # pylint: disable=import-outside-toplevel

    m = mask[:, None, :].to(z1.dtype)  # (B, 1, L)
    # cost[i, j] = Σ_l m_j (z0_i - z1_j)² = Σ m_j z0_i² - 2 Σ m_j z0_i z1_j + Σ m_j z1_j²
    z0_sq = torch.einsum("icl,jl->ij", z0.square(), m[:, 0])
    cross = torch.einsum("icl,jcl->ij", z0, z1 * m)
    z1_sq = (z1.square() * m).sum((1, 2))[None, :]
    cost = (z0_sq - 2 * cross + z1_sq).float().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty_like(cols)
    perm[cols] = rows  # utterance j gets noise sample perm[j]
    return z0[torch.as_tensor(perm, device=z0.device)]


class LatentEnergyMatching(TextConditionedLatentModel):
    def __init__(self, text_encoder: nn.Module, latent_dim: int = 32, backbone: dict | None = None,
                 duration: dict | None = None, p_uncond: float = 0.1, duration_weight: float = 1.0,
                 output_scale: float = 1.0, tau_star: float = 1.0, ot_coupling: bool = True,
                 cd_weight: float = 0.0, cd_start_step: int = 0, neg_batch: int = 8, neg_steps: int = 20,
                 neg_dt: float = 0.05,
                 neg_eps_max: float = 0.05, trim_alpha: float = 0.1, clamp_beta: float = 0.05):
        super().__init__(text_encoder, latent_dim, duration, p_uncond, duration_weight)
        self.backbone = LatentTransformer(latent_dim, text_encoder.dim, out_dim=1, time_conditioned=False,
                                          **(backbone or {}))
        self.output_scale, self.tau_star, self.ot_coupling = output_scale, tau_star, ot_coupling
        self.cd_weight, self.cd_start_step = cd_weight, cd_start_step
        self.neg_batch, self.neg_steps, self.neg_dt, self.neg_eps_max = neg_batch, neg_steps, neg_dt, neg_eps_max
        self.trim_alpha, self.clamp_beta = trim_alpha, clamp_beta
        # Optimiser steps taken; switches phase 2 on and survives resume (it is in the state dict).
        self.register_buffer("train_steps", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ energy
    def energy(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor) -> Tensor:
        """Total energy per utterance `(B,)`: sum of the per-frame outputs over valid frames."""
        e = self.backbone(z * mask[:, None, :].to(z.dtype), mask, h, h_mask)  # (B, 1, L), zero on padding
        return self.output_scale * e.sum((1, 2))

    def grad_energy(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor, create_graph: bool = False,
                    guidance_scale: float = 1.0) -> Tensor:
        """`∇_z V(z)`; with `guidance_scale ≠ 1` the CFG combination `∇V_u + w (∇V_c - ∇V_u)`,
        i.e. the gradient of the energy `V_u + w (V_c - V_u)`.

        `create_graph` (training: the loss is a function of ∇V) forces the math attention kernel,
        the only one with a double backward; plain gradients use the fast kernels."""
        kernel = sdpa_kernel(SDPBackend.MATH) if create_graph else contextlib.nullcontext()
        with torch.enable_grad(), kernel:
            if guidance_scale != 1.0:
                null, null_mask = self.null_condition(h, h_mask)
                z2 = torch.cat([z, z]).detach().requires_grad_(True)
                v = self.energy(z2, torch.cat([mask, mask]), torch.cat([h, null]), torch.cat([h_mask, null_mask]))
                g_cond, g_uncond = torch.autograd.grad(v.sum(), z2, create_graph=create_graph)[0].chunk(2)
                return g_uncond + guidance_scale * (g_cond - g_uncond)
            if not create_graph:
                z = z.detach()
            if not z.requires_grad:
                z = z.requires_grad_(True)
            return torch.autograd.grad(self.energy(z, mask, h, h_mask).sum(), z, create_graph=create_graph)[0]

    def langevin(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor, n_steps: int, dt: float, eps_fn,
                 guidance_scale: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
        """`z ← z - dt ∇V(z) + sqrt(2 dt ε(step)) ξ`, kept zero on padded frames."""
        m = mask[:, None, :].to(z.dtype)
        z = z.detach() * m
        for step in range(n_steps):
            g = self.grad_energy(z, mask, h, h_mask, guidance_scale=guidance_scale).detach()
            z = z - dt * g
            eps = eps_fn(step)
            if eps > 0:
                noise = torch.randn(z.shape, generator=generator, device="cpu" if generator is not None else z.device)
                z = z + math.sqrt(2 * dt * eps) * noise.to(z.device, z.dtype)
            z = z * m
        return z.detach()

    # ------------------------------------------------------------------ training
    def ot_loss(self, z1: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor,
                noise: Tensor | None = None, t: Tensor | None = None, create_graph: bool = True) -> Tensor:
        m = mask[:, None, :].to(z1.dtype)
        z0 = torch.randn_like(z1) if noise is None else noise
        if self.ot_coupling:
            z0 = ot_couple(z0, z1, mask)
        if t is None:
            t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype) * self.tau_star
        tt = t[:, None, None]
        zt = ((1 - tt) * z0 + tt * z1) * m
        grad = self.grad_energy(zt, mask, h, h_mask, create_graph=create_graph)
        return ((grad - (z0 - z1)).square() * m).sum() / (m.sum() * self.latent_dim)

    def cd_loss(self, z1: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor) -> tuple[Tensor, dict]:
        """CD on the first `neg_batch` utterances (each negative costs `neg_steps` forward and
        backward passes); chains run without dropout, so they follow the field being trained."""
        n = min(self.neg_batch, z1.shape[0])
        z1, mask, h, h_mask = z1[:n], mask[:n], h[:n], h_mask[:n]
        lengths = mask.sum(1).to(z1.dtype)
        half = n // 2
        was_training = self.training
        self.eval()
        neg_data = self.langevin(z1[:half], mask[:half], h[:half], h_mask[:half], self.neg_steps, self.neg_dt,
                                 lambda _: self.neg_eps_max)
        neg_noise = self.langevin(torch.randn_like(z1[half:]), mask[half:], h[half:], h_mask[half:], self.neg_steps,
                                  self.neg_dt, lambda s: eps_schedule(s * self.neg_dt, self.neg_eps_max, self.tau_star))
        neg = torch.cat([neg_data, neg_noise])
        self.train(was_training)
        v_data = self.energy(z1, mask, h, h_mask) / lengths
        v_neg = self.energy(neg, mask, h, h_mask) / lengths
        cd = (v_data.mean() - trimmed_mean(v_neg, self.trim_alpha)) / self.neg_eps_max
        stats = {"v_data": v_data.mean().detach(), "v_neg": v_neg.mean().detach()}
        return torch.clamp(cd, min=-self.clamp_beta), stats

    @property
    def cd_active(self) -> bool:
        return self.cd_weight > 0 and int(self.train_steps) >= self.cd_start_step

    def training_step(self, batch: dict, noise: Tensor | None = None, t: Tensor | None = None,
                      cond_drop: bool = True) -> dict:
        z, z_lengths = batch["z"], batch["z_lengths"]
        mask = sequence_mask(z_lengths, z.shape[-1])
        h, h_mask = self.encode_text(batch["texts"], z.device)
        dur_loss = self.duration_loss(h, h_mask, z_lengths)
        if cond_drop:
            h, h_mask = self.random_drop(h, h_mask)

        ot = self.ot_loss(z, mask, h, h_mask, noise, t)
        loss = ot + self.duration_weight * dur_loss
        out = {"ot": ot, "dur": dur_loss}
        if self.cd_active:
            cd, stats = self.cd_loss(z, mask, h, h_mask)
            loss = loss + self.cd_weight * cd
            out.update(cd=cd, **stats)
        if self.training:
            self.train_steps += 1
        return {"loss": loss, **out}

    def validation_losses(self, z: Tensor, mask: Tensor, h: Tensor, h_mask: Tensor, noise: Tensor, t: Tensor) -> dict:
        """Per-frame validation metrics at fixed noise and times: the OT loss, and the energy per
        frame of the data and of pure noise (after phase 2 the data should sit clearly lower)."""
        lengths = mask.sum(1).to(z.dtype)
        m = mask[:, None, :].to(z.dtype)
        with torch.no_grad():
            v_data = (self.energy(z, mask, h, h_mask) / lengths).mean()
            v_noise = (self.energy(noise * m, mask, h, h_mask) / lengths).mean()
        return {"ot": self.ot_loss(z, mask, h, h_mask, noise, t * self.tau_star, create_graph=False),
                "v_data": v_data, "v_noise": v_noise}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def synthesise(self, texts: list[str], dt: float = 0.05, tau_s: float = 1.0, eps_max: float = 0.0,
                   guidance_scale: float = 2.0, temperature: float = 1.0, length_scale: float = 1.0,
                   z_lengths: Tensor | None = None, generator: torch.Generator | None = None) -> dict:
        """Texts -> normalised latents `z (B, C, L)` and `z_lengths (B,)` by Langevin dynamics on V.

        `tau_s = tau_star, eps_max = 0` integrates the ODE `dz/dτ = -∇V` (all a phase-1 model
        supports); after phase 2, `tau_s > tau_star` with `eps_max > 0` adds Langevin refinement
        around the data minima. `z_lengths` overrides the length predictor."""
        h, h_mask = self.encode_text(texts, self.device)
        if z_lengths is None:
            z_lengths = self.predict_lengths(h, h_mask, length_scale)
        z_lengths = z_lengths.to(self.device)
        mask = sequence_mask(z_lengths, int(z_lengths.max()))
        z = self.initial_noise(len(texts), mask.shape[1], generator, h.dtype) * temperature
        n_steps = int(round(tau_s / dt))
        z = self.langevin(z, mask, h, h_mask, n_steps, dt, lambda s: eps_schedule(s * dt, eps_max, self.tau_star),
                          guidance_scale, generator)
        return {"z": z, "z_lengths": z_lengths}
