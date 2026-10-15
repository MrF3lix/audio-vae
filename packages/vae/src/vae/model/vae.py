import torch
from torch import nn, Tensor

from vae.model.encoder import Encoder
from vae.model.decoder import Decoder
from common.lengths import sequence_mask

class VAE(nn.Module):

    def __init__(
            self,
            n_mels: int = 96,
            latent_dim: int = 32,
            compression: int = 4,
            channels: list[int] = (192, 256, 384),
            blocks_per_level: int = 2,
            kernel_size: int = 7,
            beta: float = 5e-3
        ):
        super().__init__()

        self.encoder = Encoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)
        self.decoder = Decoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)

        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.compression = compression
        self.beta = beta
        self.n_levels = len(channels)
        
        self.register_buffer("z_mean", torch.zeros(latent_dim, 1))
        self.register_buffer("z_std", torch.ones(latent_dim, 1))
        self.register_buffer("latent_stats_fitted", torch.tensor(False))

    def level_masks(self, lengths: Tensor | None, t_max: int) -> list[Tensor] | None:
        if lengths is None:
            return None
        if t_max % self.compression:
            raise ValueError(f"mel length {t_max} must be a multiple of compression={self.compression}")
        masks = []
        for level in range(self.n_levels):
            factor = 2**level
            masks.append(sequence_mask(-(-lengths // factor), t_max // factor).unsqueeze(1).to(torch.float32))
        return masks

    def normalize_latent(self, z: Tensor) -> Tensor:
        return (z - self.z_mean) / self.z_std

    def denormalize_latent(self, z: Tensor) -> Tensor:
        return z * self.z_std + self.z_mean

    def encode(self, mel: Tensor, lengths: Tensor | None = None, normalize: bool = False):
        masks = self.level_masks(lengths, mel.shape[-1])
        mu, logvar = self.encoder(mel, masks)
        if normalize:
            mu = self.normalize_latent(mu)
            logvar = logvar - 2 * torch.log(self.z_std)

        return mu, logvar

    def decode(self, z: Tensor, z_lengths: Tensor | None = None, normalized: bool = False):
        if normalized:
            z = self.denormalize_latent(z)
        lengths = None if z_lengths is None else z_lengths * self.compression
        masks = self.level_masks(lengths, z.shape[-1] * self.compression)

        return self.decoder(z, masks)

    def forward(self, mel: Tensor, lengths: Tensor | None = None, sample: bool = True) -> dict:
        mu, logvar = self.encode(mel, lengths)
        z = self.reparameterize(mu, logvar) if sample else mu
        z_lengths = None if lengths is None else self.latent_lengths(lengths)
        mel_hat = self.decode(z, z_lengths)

        if lengths is None:
            recon = (mel_hat - mel).abs().mean()
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
        else:
            frame_mask = sequence_mask(lengths, mel.shape[-1]).unsqueeze(1).to(mel.dtype)
            recon = ((mel_hat - mel).abs() * frame_mask).sum() / (frame_mask.sum() * self.n_mels)

            z_mask = sequence_mask(z_lengths, mu.shape[-1]).unsqueeze(1).to(mu.dtype)
            kl_elem = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
            kl = (kl_elem * z_mask).sum() / (z_mask.sum() * self.latent_dim)

        loss = recon + self.beta * kl
        return {
            'mel_hat': mel_hat,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'loss': loss,
            'recon': recon,
            'kl': kl
        }

    def latent_lengths(self, lengths: Tensor) -> Tensor:
        return -(-lengths // self.compression)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def training_step(self, batch: dict) -> dict:
        out = self(batch['mel'], batch['lengths'])
        return {k: out[k] for k in ('loss', 'recon', 'kl')}