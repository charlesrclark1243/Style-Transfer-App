from modules.vgg_encoder import VGGEncoder
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.adain import AdaIN
from modules.style_attention import StyleAttention
from modules.style_modulation import StyleModulation, style_code

from functools import partial
import torch.nn as nn
import torch

class StyleTransferModel(nn.Module):
    """Frozen VGG encoder with AdaIN at relu4_1."""
    def __init__(self):
        super(StyleTransferModel, self).__init__()

        self.encoder = VGGEncoder()
        self.adain = AdaIN()
        self.decoder = Decoder(in_channels=512)

    def transfer(self, content_image, style_image):
        """Returns the stylized image (unbounded, clip to [0, 1] for display) and the AdaIN target features it was decoded from."""
        # The encoder is frozen, so no graph is needed through it
        with torch.no_grad():
            content_features = self.encoder(content_image)[-1]
            style_features = self.encoder(style_image)[-1]

        target_features = self.adain(content_features, style_features)
        stylized_image = self.decoder(target_features)

        return stylized_image, target_features

    def forward(self, content_image, style_image):
        return self.transfer(content_image, style_image)[0]

class ScratchAttentionModel(nn.Module):
    """
    The from-scratch two-headed encoder, with cross-attention from content to style features in place of AdaIN.

    quarter_attention adds a second attention block at 1/4 resolution, added into the decoder, so fine texture can come
    from the style image. style_modulation conditions every decoder block on a style code (FiLM), so the layers that
    paint strokes know which style they're painting. Both start as no-ops, so a checkpoint without them resumes with
    unchanged outputs.
    """
    def __init__(self, quarter_attention: bool = False, style_modulation: bool = False):
        super(ScratchAttentionModel, self).__init__()

        self.encoder = Encoder()
        self.attention = StyleAttention(channels=1024)
        self.quarter_attention = StyleAttention(channels=512, num_heads=8, residual=False) if quarter_attention else None
        self.decoder = Decoder(in_channels=1024)
        self.style_modulation = StyleModulation(style_dim=2 * 1024, channels=self.decoder.modulated_channels) if style_modulation else None

    def transfer(self, content_image, style_image):
        """Returns the stylized image (unbounded, clip to [0, 1] for display) and None, since there is no AdaIN target."""
        if self.quarter_attention is not None:
            content_features, style_features, content_quarter, style_quarter = self.encoder(
                content_image, style_image, return_quarter_features=True
            )
            quarter_features = self.quarter_attention(content_quarter, style_quarter)
        else:
            content_features, style_features = self.encoder(content_image, style_image)
            quarter_features = None

        modulation = self.style_modulation(style_code(style_features)) if self.style_modulation is not None else None

        stylized_image = self.decoder(
            self.attention(content_features, style_features),
            quarter_resolution_features=quarter_features,
            style_modulation=modulation,
        )

        return stylized_image, None

    def forward(self, content_image, style_image):
        return self.transfer(content_image, style_image)[0]

MODELS = {
    'vgg-adain': StyleTransferModel,
    'scratch-attention': ScratchAttentionModel,
    'scratch-attention-multiscale': partial(ScratchAttentionModel, quarter_attention=True),
    'scratch-attention-styled': partial(ScratchAttentionModel, quarter_attention=True, style_modulation=True),
}

def read_checkpoint(checkpoint: dict):
    """Returns (model name, state dict). Checkpoints record their model; bare state dicts predate that and are the VGG AdaIN model."""
    if "state_dict" in checkpoint:
        return checkpoint["model"], checkpoint["state_dict"]
    return "vgg-adain", checkpoint
