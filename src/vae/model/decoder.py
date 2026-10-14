import torch.nn.functional as F
from torch import nn, Tensor

from vae.model.shared_modules import Stage, ChannelLayerNorm

class Decoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            latent_dim: int,
            channels: list[int],
            blocks_per_level: int,
            kernel_size: int
        ):
        super().__init__()

        rev_channels = list(reversed(channels))
        self.conv_in = nn.Conv1d(latent_dim, rev_channels[0], 1)
        self.stages = nn.ModuleList([Stage(c, blocks_per_level, kernel_size) for c in rev_channels])
        self.ups = nn.ModuleList([Upsample(rev_channels[i], rev_channels[i+1]) for i in range(len(rev_channels) - 1)])
        self.norm_out = ChannelLayerNorm(rev_channels[-1])
        self.conv_out = nn.Conv1d(rev_channels[-1], n_mels, kernel_size, padding=kernel_size // 2)

    def forward(self, z: Tensor, masks: list[Tensor] | None) -> Tensor:
        x = self.conv_in(z)
        rev_masks = list(reversed(masks)) if masks is not None else None

        for i, stage in enumerate(self.stages):
            mask = rev_masks[i] if rev_masks is not None else None
            x = stage(x * mask if mask is not None else x, mask)
            if i < len(self.ups):
                x = self.ups[i](x)

        mask = masks[0] if masks is not None else None
        out = self.conv_out(F.gelu(self.norm_out(x)))
        return out * mask if mask is not None else out


class Upsample(nn.Module):
    def __init__(
            self,
            channels_in: int,
            channels_out: int
    ):
        super().__init__()

        self.norm = ChannelLayerNorm(channels_in)
        self.conv = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x))