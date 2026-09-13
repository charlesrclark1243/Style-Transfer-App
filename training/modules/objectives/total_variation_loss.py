import torch.nn as nn
import torch

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

        return vertical.mean() + horizontal.mean()