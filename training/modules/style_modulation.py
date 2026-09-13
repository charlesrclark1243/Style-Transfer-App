import torch.nn as nn
import torch

def style_code(style_features):
    """Summarizes style features as each channel's mean and std: one vector per image, whatever the image size."""
    with torch.autocast(device_type=style_features.device.type, enabled=False):
        style_features = style_features.float()
        return torch.cat([style_features.mean(dim=[2, 3]), style_features.std(dim=[2, 3])], dim=1)

class StyleModulation(nn.Module):
    """
    Maps a style code to a per-channel scale and shift for each modulated decoder layer (FiLM), so every decoder block,
    down to full resolution, is conditioned on the style. Starts as a no-op: scale 1, shift 0.
    """
    def __init__(self, style_dim: int, channels: list[int], hidden_dim: int = 512):
        super(StyleModulation, self).__init__()

        self.channels = list(channels)

        self.mlp = nn.Sequential(
            # Normalizing the code keeps the MLP stable whatever the scale of the encoder's features
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2 * sum(self.channels))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, code):
        """Returns a list of (scale, shift) pairs shaped B × C × 1 × 1, one per entry in channels."""
        scales, shifts = self.head(self.mlp(code)).chunk(2, dim=1)

        return [
            (1 + scale[..., None, None], shift[..., None, None])
            for scale, shift in zip(scales.split(self.channels, dim=1), shifts.split(self.channels, dim=1))
        ]
