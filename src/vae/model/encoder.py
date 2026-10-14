from torch import nn, Tensor

from vae.model.shared_modules import Stage, ChannelLayerNorm

class Encoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            latent_dim: int,
            channels: list[int],
            blocks_per_level: int,
            kernel_size: int
        ):
        super().__init__()


        self.conv_in = nn.Conv1d(n_mels, channels[0], kernel_size, padding=kernel_size // 2)
        self.stages = nn.ModuleList([Stage(c, blocks_per_level, kernel_size) for c in channels])
        self.downs = nn.ModuleList([Downsample(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
        self.norm_out = ChannelLayerNorm(channels[-1])
        self.head = nn.Conv1d(channels[-1], 2 * latent_dim, 1)


    def forward(self, mel: Tensor, masks: list[Tensor] | None) -> tuple[Tensor, Tensor]:
        x = self.conv_in(mel)
        for level, stage in enumerate(self.stages):
            mask = masks[level] if masks is not None else None
            x = stage(x * mask if mask is not None else x, mask)
            if level < len(self.downs):
                x = self.downs[level](x)

        mask = masks[-1] if masks is not None else None
        out = self.head(self.norm_out(x))

        if mask is not None:
            out = out * mask

        mu, logvar = out.chunk(2, dim=1)

        return mu, logvar

class Downsample(nn.Module):
    def __init__(
            self,
            channels_in: int,
            channels_out: int
        ):
        super().__init__()

        self.norm = ChannelLayerNorm(channels_in)
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x))
