import torch
from torch import Tensor

def sequence_mask(lenghts: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lenghts.max())
    positions = torch.arange(max_length, device=lenghts.device)
    return positions.unsqueeze(0) < lenghts.unsqueeze(1)