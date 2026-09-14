import torch
from torch import nn


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

        return vertical.mean() + horizontal.mean()
