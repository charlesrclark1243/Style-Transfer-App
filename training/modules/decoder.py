import torch
from torch import nn


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    # Reflection padding avoids the frame that zero padding leaves around the output
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.ReLU(),
            conv3x3(in_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Decoder(nn.Module):
    """Decodes relu4_1 features back to an image. The output is unbounded (no sigmoid, as in AdaIN), so it can't saturate during training."""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            conv3x3(512, 512),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv3x3(512, 512),
            nn.ReLU(),
            conv3x3(512, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv3x3(256, 256),
            nn.ReLU(),
            conv3x3(256, 128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv3x3(128, 64),
            nn.ReLU(),
            conv3x3(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        return x
