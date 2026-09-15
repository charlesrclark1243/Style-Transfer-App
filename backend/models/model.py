import torch
from torch import nn

from models.adain import AdaIN
from models.decoder import Decoder
from models.vgg_encoder import VGGEncoder


class StyleTransferModel(nn.Module):
    """Frozen VGG encoder with AdaIN at relu4_1."""

    def __init__(self):
        super().__init__()

        self.encoder = VGGEncoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def transfer(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        alpha: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the stylized image (unbounded, clip to [0, 1] for display) and the AdaIN target features it was decoded from.

        alpha sets the style strength, from 0 (the content's own features) to 1 (the full AdaIN output).
        """
        # The encoder is frozen, so no graph is needed through it
        with torch.no_grad():
            content_features = self.encoder(content_image)[-1]
            style_features = self.encoder(style_image)[-1]

        target_features = self.adain(content_features, style_features)
        # Blends in feature space, as in the AdaIN paper; blending output pixels would just cross-fade two images
        target_features = alpha * target_features + (1 - alpha) * content_features
        stylized_image = self.decoder(target_features)

        return stylized_image, target_features

    def forward(self, content_image, style_image, alpha: float = 1.0):
        return self.transfer(content_image, style_image, alpha)[0]


def read_checkpoint(checkpoint: dict) -> dict:
    """Returns the model state dict from a checkpoint saved by main_loop."""
    return checkpoint["state_dict"]
