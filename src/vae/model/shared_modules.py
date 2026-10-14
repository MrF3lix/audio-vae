import torch
import torch.nn.functional as F
from torch import nn, Tensor

class ChannelLayerNorm(nn.Module):
    def __init__(
            self,
            channels: int,
            eps: float = 1e-6
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ConvNeXtBlock1d(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            mlp_ratio: int = 4,
            layer_scale: float = 1e-6
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.norm = ChannelLayerNorm(channels)
        self.pw1 = nn.Conv1d(channels, mlp_ratio * channels, 1)
        self.pw2 = nn.Conv1d(mlp_ratio * channels, channels, 1)

        self.gamma = nn.Parameter(layer_scale * torch.ones(channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pw2(F.gelu(self.pw1(h)))

        return x + self.gamma * h

class Stage(nn.Module):
    def __init__(
            self,
            channels: int,
            n_blocks: int,
            kernel_size: int
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[ConvNeXtBlock1d(channels, kernel_size) for _ in range(n_blocks)])

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = self.blocks(x)
        return x * mask if mask is not None else x