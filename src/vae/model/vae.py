import torch
from torch import nn, Tensor

from vae.model.encoder import Encoder
from vae.model.decoder import Decoder
from vae.model.lengths import sequence_mask

class VAE(nn.Module):

    def __init__(
            self,
            n_mels: int = 96,
            latent_dim: int = 32,
            channels: list[int] = (192, 256, 384),
            blocks_per_level: int = 2,
            kernel_size: int = 7,
            beta: float = 5e-3
        ):
        super().__init__()

        self.encoder = Encoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)
        self.decoder = Decoder(n_mels, latent_dim, channels, blocks_per_level, kernel_size)


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

        return

    def latent_lengths(self, lengths: Tensor) -> Tensor:
        return -(-lengths // self.compression)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)