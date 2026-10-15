"""Public entry point for using a trained VAE from other packages.

    from vae import load_pretrained
    vae = load_pretrained("logs/train_vae/.../best.pt", device="mps")
    z, _ = vae.encode(vae.normalize_mel(mel)[None], lengths)        # normalised latents
    mel = vae.denormalize_mel(vae.decode(z, z_lengths))             # vocoder feature space

Everything a downstream model needs (latent size, compression, mel front-end and statistics)
comes from the checkpoint, so callers never compose the VAE's hydra configs themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from torch import Tensor

from common.training.utils import load_checkpoint
from vae.model.vae import VAE

# Checkpoints written before the split into `common` / `vae` reference the old module paths.
_LEGACY_PREFIXES = {"vae.audio.": "common.audio.", "vae.data.": "common.data.", "vae.training.": "common.training."}


def upgrade_cfg(cfg):
    """Rewrite `_target_`s of pre-split checkpoints (plain dict/list config, in place)."""
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == "_target_" and isinstance(value, str):
                for old, new in _LEGACY_PREFIXES.items():
                    if value.startswith(old):
                        cfg[key] = new + value[len(old):]
            else:
                upgrade_cfg(value)
    elif isinstance(cfg, list):
        for value in cfg:
            upgrade_cfg(value)
    return cfg


@dataclass
class PretrainedVAE:
    model: VAE
    mel: torch.nn.Module  # WhisperMel front-end the VAE was trained on
    mel_stats: dict  # {"mean", "std"} scalars the mels were normalised with
    ckpt_path: Path
    step: int

    @property
    def latent_dim(self) -> int:
        return self.model.latent_dim

    @property
    def compression(self) -> int:
        return self.model.compression

    @property
    def device(self) -> torch.device:
        return self.model.z_mean.device

    def latent_lengths(self, mel_lengths: Tensor) -> Tensor:
        return self.model.latent_lengths(mel_lengths)

    def normalize_mel(self, mel: Tensor) -> Tensor:
        return (mel - self.mel_stats["mean"]) / self.mel_stats["std"]

    def denormalize_mel(self, mel: Tensor) -> Tensor:
        return mel * self.mel_stats["std"] + self.mel_stats["mean"]

    @torch.no_grad()
    def encode(self, mel: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Normalised mel `(B, n_mels, T)`, `T` a multiple of `compression` -> normalised `(mu, logvar)`."""
        return self.model.encode(mel, lengths, normalize=True)

    @torch.no_grad()
    def decode(self, z: Tensor, z_lengths: Tensor | None = None) -> Tensor:
        """Normalised latents `(B, C, L)` -> normalised mel `(B, n_mels, L * compression)`."""
        return self.model.decode(z, z_lengths, normalized=True)


def load_pretrained(ckpt_path: str | Path, device="cpu", use_ema: bool = True,
                    require_latent_stats: bool = True) -> PretrainedVAE:
    """Frozen VAE in eval mode, plus its mel front-end and statistics."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path)
    cfg = upgrade_cfg(ckpt["cfg"])

    model = hydra.utils.instantiate(cfg["model"])
    model.load_state_dict(ckpt["ema"] if use_ema and ckpt.get("ema") is not None else ckpt["model"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if require_latent_stats and not bool(model.latent_stats_fitted):
        raise RuntimeError(f"{ckpt_path} has no latent statistics; run: uv run vae-latent-stats --ckpt {ckpt_path}")

    mel = hydra.utils.instantiate(cfg["data"]["mel"])
    mel_stats = ckpt.get("mel_stats") or cfg["data"]["mel_stats"]
    return PretrainedVAE(model, mel, {k: float(v) for k, v in mel_stats.items()}, ckpt_path, int(ckpt["step"]))
