import torch.nn.functional as F
import torch.nn as nn
import torch

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, stylized_features, style_features):
        stylized_mean = torch.mean(stylized_features, dim=[2, 3], keepdim=True)
        stylized_std = torch.std(stylized_features, dim=[2, 3], keepdim=True)

        style_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
        style_std = torch.std(style_features, dim=[2, 3], keepdim=True)

        mean_loss = F.mse_loss(stylized_mean, style_mean)
        std_loss = F.mse_loss(stylized_std, style_std)

        return mean_loss + std_loss