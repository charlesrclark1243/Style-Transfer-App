import torch.nn as nn
import torch

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_features, style_features):
        content_mean = torch.mean(content_features, dim=[2, 3], keepdim=True)
        content_std = torch.std(content_features, dim=[2, 3], keepdim=True)

        style_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
        style_std = torch.std(style_features, dim=[2, 3], keepdim=True)

        normalized_content = (content_features - content_mean) / (content_std + 1e-5)
        stylized_features = normalized_content * style_std + style_mean

        return stylized_features