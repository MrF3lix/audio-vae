"""Paired (text, latent) data for the text-conditioned generative models.

Latents come from the VAE's latent cache (`vae.latents`), built on first use for every
utterance of both filelists. Whole utterances only, never crops: the text must match the audio.
Batches group utterances of similar length, so little compute is spent on padding.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from common.data.filelist import parse_filelist
from common.training.utils import resolve_device
from vae import load_pretrained
from vae.latents import ensure_latent_cache

log = logging.getLogger(__name__)


class LatentTextDataset(Dataset):
    """Returns `{"z": (C, L) normalised latent, "text": str, "path": str}`.

    With `sample_latent`, `z` is a posterior sample instead of the posterior mean (augmentation)."""

    def __init__(self, filelist, latent_dir: str | Path, root=None, sample_latent: bool = False):
        self.items = parse_filelist(filelist, root)
        self.latent_dir = Path(latent_dir)
        self.sample_latent = sample_latent
        self.lengths = [np.load(self.latent_dir / f"{u.path.stem}.npy", mmap_mode="r").shape[-1] for u in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        utt = self.items[index]
        lat = torch.from_numpy(np.load(self.latent_dir / f"{utt.path.stem}.npy"))
        mu, logvar = lat[0], lat[1]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.sample_latent else mu
        return {"z": z, "text": utt.text, "path": str(utt.path)}


def collate_latents(batch: list[dict]) -> dict:
    """Zero-pad to the longest latent: `z (B, C, L)`, `z_lengths (B,)`, `texts`, `paths`."""
    z_lengths = torch.tensor([b["z"].shape[-1] for b in batch], dtype=torch.long)
    z = torch.zeros(len(batch), batch[0]["z"].shape[0], int(z_lengths.max()))
    for i, b in enumerate(batch):
        z[i, :, : z_lengths[i]] = b["z"]
    return {"z": z, "z_lengths": z_lengths, "texts": [b["text"] for b in batch], "paths": [b["path"] for b in batch]}


class BucketBatchSampler(Sampler):
    """Shuffle, cut into pools of `pool_batches` batches, sort each pool by length, batch it,
    shuffle the batches. A new permutation every epoch."""

    def __init__(self, lengths: list[int], batch_size: int, pool_batches: int = 50, shuffle: bool = True,
                 drop_last: bool = True, seed: int = 0):
        self.lengths, self.batch_size, self.pool = lengths, batch_size, pool_batches * batch_size
        self.shuffle, self.drop_last, self.seed, self.epoch = shuffle, drop_last, seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.pool):
            pool = sorted(idx[start : start + self.pool], key=lambda i: self.lengths[i])
            batches += [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.lengths)
        if not self.drop_last:
            return sum(-(-min(self.pool, n - s) // self.batch_size) for s in range(0, n, self.pool))
        return sum(min(self.pool, n - s) // self.batch_size for s in range(0, n, self.pool))


class LatentTextDataModule:
    """Loads the VAE (frozen; also used by validation to decode samples), makes sure the latent
    cache is complete, and builds the loaders."""

    def __init__(self, name: str, train_filelist: str, valid_filelist: str, vae_ckpt: str, latent_cache_root: str,
                 batch_size: int, num_workers: int, pin_memory: bool, mel_cache_dir: str | None = None,
                 sample_latent: bool = False, pool_batches: int = 50, root: str | None = None, seed: int = 0,
                 device: str = "auto"):
        self.name, self.batch_size, self.num_workers, self.pin_memory = name, batch_size, num_workers, pin_memory
        self.pool_batches, self.seed = pool_batches, seed
        self.vae = load_pretrained(vae_ckpt, resolve_device(device))
        self.latent_dir, self.latent_meta = ensure_latent_cache(
            self.vae, [train_filelist, valid_filelist], latent_cache_root, mel_cache_dir, root
        )
        log.info("Latents: %s", self.latent_dir)
        self.train_set = LatentTextDataset(train_filelist, self.latent_dir, root, sample_latent)
        self.valid_set = LatentTextDataset(valid_filelist, self.latent_dir, root, False)

    @property
    def latent_dim(self) -> int:
        return self.vae.latent_dim

    def train_dataloader(self):
        sampler = BucketBatchSampler(self.train_set.lengths, self.batch_size, self.pool_batches, True, True, self.seed)
        return DataLoader(self.train_set, batch_sampler=sampler, num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0, collate_fn=collate_latents)

    def valid_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=self.pin_memory, collate_fn=collate_latents)
