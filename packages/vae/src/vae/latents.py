"""Cache of VAE latents for downstream generative models.

One `.npy` per utterance holding `(2, C, L)` = posterior mean and log-variance, both in the
*normalised* latent space (`encode(..., normalize=True)`), plus `meta.json` with the VAE
identity, latent and mel statistics, and every utterance's mel frame count.

The directory is keyed by VAE run, checkpoint and step (`<run>_<ckpt>_step<k>`), so latents of
a retrained or resumed VAE are never mixed with older ones.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.data.mel_dataset import MelDataset
from common.lengths import pad_to_multiple
from vae.pretrained import PretrainedVAE

log = logging.getLogger(__name__)


def cache_key(vae: PretrainedVAE) -> str:
    path = vae.ckpt_path
    run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return f"{run}_{path.stem}_step{vae.step}"


@torch.no_grad()
def ensure_latent_cache(vae: PretrainedVAE, filelists: list[str | Path], cache_root: str | Path,
                        mel_cache_dir: str | Path | None = None, root: str | Path | None = None) -> tuple[Path, dict]:
    """Encode every utterance of `filelists` that is not cached yet; returns `(dir, meta)`.

    `mel_cache_dir` is the VAE stage's mel cache (shared, so mels are extracted only once)."""
    out_dir = Path(cache_root) / cache_key(vae)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    lengths = meta.get("lengths", {})

    todo = []
    for filelist in filelists:
        ds = MelDataset(filelist, vae.mel, vae.mel_stats, None, 1, mel_cache_dir, root)
        todo += [(ds, i) for i, u in enumerate(ds.items)
                 if u.path.stem not in lengths or not (out_dir / f"{u.path.stem}.npy").is_file()]
    if todo:
        log.info("Encoding %d utterances with %s -> %s", len(todo), vae.ckpt_path, out_dir)
        r = vae.compression
        for ds, i in todo:
            stem = ds.items[i].path.stem
            mel = ds.normalize(ds.raw_mel(i))
            T = mel.shape[-1]
            x = F.pad(mel, (0, pad_to_multiple(T, r) - T))[None].to(vae.device)
            mu, logvar = vae.encode(x, torch.tensor([T], device=vae.device))
            L = -(-T // r)
            lat = torch.stack([mu[0, :, :L], logvar[0, :, :L]]).cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{stem}.npy", lat)
            lengths[stem] = T

    meta = {
        "vae_ckpt": str(vae.ckpt_path.resolve()),
        "vae_step": vae.step,
        "latent_dim": vae.latent_dim,
        "compression": vae.compression,
        "mel_stats": vae.mel_stats,
        "z_mean": vae.model.z_mean.flatten().tolist(),
        "z_std": vae.model.z_std.flatten().tolist(),
        "lengths": lengths,
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return out_dir, meta
