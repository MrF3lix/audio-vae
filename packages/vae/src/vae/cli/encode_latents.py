"""Encode filelists with a trained VAE into the latent cache (`vae.latents`).

    uv run vae-encode-latents --ckpt logs/train_vae/vae_be/runs/<date>/checkpoints/best.pt \
        --filelists data/filelists/be/train.txt data/filelists/be/val.txt

Only utterances missing from the cache are encoded, so this is cheap to re-run. The generative
models call the same code on start-up; this command exists to build the cache ahead of time
(e.g. on a login node) and to check the round trip with `--check`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from common.training.utils import resolve_device
from vae.latents import ensure_latent_cache
from vae.pretrained import load_pretrained


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--filelists", type=Path, nargs="+", required=True)
    p.add_argument("--cache-root", type=Path, default=Path("data/cache/latents"))
    p.add_argument("--mel-cache-dir", type=Path, default=None,
                   help="mel cache of the VAE stage (default: data.cache_dir from the checkpoint config)")
    p.add_argument("--root", type=Path, default=Path("."), help="base for relative filelist paths")
    p.add_argument("--device", default="auto")
    p.add_argument("--check", type=int, default=0, metavar="N",
                   help="decode N cached latents and report the L1 to the original mel")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    vae = load_pretrained(args.ckpt, resolve_device(args.device))
    mel_cache_dir = args.mel_cache_dir
    if mel_cache_dir is None:
        from common.training.utils import load_checkpoint  # pylint: disable=import-outside-toplevel

        mel_cache_dir = load_checkpoint(args.ckpt)["cfg"]["data"].get("cache_dir")
    out_dir, meta = ensure_latent_cache(vae, args.filelists, args.cache_root, mel_cache_dir, args.root)
    print(f"{len(meta['lengths'])} utterances cached in {out_dir}")

    if args.check:
        from common.data.mel_dataset import MelDataset  # pylint: disable=import-outside-toplevel

        ds = MelDataset(args.filelists[0], vae.mel, vae.mel_stats, None, 1, mel_cache_dir, args.root)
        errors = []
        for i in range(min(args.check, len(ds))):
            mel = ds.normalize(ds.raw_mel(i))
            lat = torch.from_numpy(np.load(out_dir / f"{ds.items[i].path.stem}.npy"))
            z = lat[0][None].to(vae.device)
            mel_hat = vae.decode(z, torch.tensor([z.shape[-1]], device=vae.device))[0, :, : mel.shape[-1]].cpu()
            errors.append(float((mel_hat - mel).abs().mean()))
        print(f"round trip L1 (normalised mel) over {len(errors)} utterances: {np.mean(errors):.4f}")


if __name__ == "__main__":
    main()
