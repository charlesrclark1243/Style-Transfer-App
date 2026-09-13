import torch.nn as nn
import torch

def conv3x3(in_channels: int, out_channels: int):
    # Reflection padding avoids the frame that zero padding leaves around the output
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.ReLU(),
            conv3x3(in_channels, in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    """Decodes features at 1/8 resolution back to an image. The output is unbounded (no sigmoid, as in AdaIN), so it can't saturate during training."""
    def __init__(self, in_channels: int = 512):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            conv3x3(in_channels, 512),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(512, 512),
            nn.ReLU(),
            conv3x3(512, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(256, 256),
            nn.ReLU(),
            conv3x3(256, 128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(128, 64),
            nn.ReLU(),
            conv3x3(64, 3),
        )

        # Outputs of every ReLU and residual block can be modulated by a style; maps layer index to its channel count
        self.modulated_layers = {}
        channels = in_channels
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                channels = layer.out_channels
            if isinstance(layer, (nn.ReLU, ResidualBlock)):
                self.modulated_layers[index] = channels

        self.first_upsample = next(index for index, layer in enumerate(self.layers) if isinstance(layer, nn.Upsample))

    @property
    def modulated_channels(self):
        return list(self.modulated_layers.values())

    def forward(self, x, quarter_resolution_features=None, style_modulation=None):
        """
        quarter_resolution_features are added right after the first upsample, where the decoder is at 1/4 resolution with 512 channels.
        style_modulation is a list of (scale, shift) pairs, one per entry in modulated_layers, applied to those layers' outputs.
        """
        modulations = iter(style_modulation) if style_modulation is not None else None

        for index, layer in enumerate(self.layers):
            x = layer(x)

            if modulations is not None and index in self.modulated_layers:
                scale, shift = next(modulations)
                x = torch.addcmul(shift, x, scale)

            if quarter_resolution_features is not None and index == self.first_upsample:
                x = x + quarter_resolution_features

        return x
