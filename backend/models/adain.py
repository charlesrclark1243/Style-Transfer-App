import torch
from torch import nn


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, content_features: torch.Tensor, style_features: torch.Tensor
    ) -> torch.Tensor:
        # Normalization runs in float32 even under autocast; bfloat16 is too coarse for stds and dividing by them
        with torch.autocast(device_type=content_features.device.type, enabled=False):
            content_features = content_features.float()
            style_features = style_features.float()

            content_mean = torch.mean(content_features, dim=[2, 3], keepdim=True)
            content_std = torch.std(content_features, dim=[2, 3], keepdim=True)

            style_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
            style_std = torch.std(style_features, dim=[2, 3], keepdim=True)

            normalized_content = (content_features - content_mean) / (
                content_std + 1e-5
            )
            stylized_features = normalized_content * style_std + style_mean

        return stylized_features
