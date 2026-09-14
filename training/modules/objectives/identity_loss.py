import torch
import torch.nn.functional as F
from torch import nn


class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, generated_image: torch.Tensor, target_image: torch.Tensor
    ) -> torch.Tensor:
        identity_loss = F.l1_loss(generated_image, target_image)
        return identity_loss
