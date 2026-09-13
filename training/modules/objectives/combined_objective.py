from modules.vgg_encoder import VGGEncoder
from modules.style_attention import mean_variance_norm
from modules.objectives.style_loss import StyleLoss
from modules.objectives.identity_loss import IdentityLoss
from modules.objectives.total_variation_loss import TotalVariationLoss

import torch.nn.functional as F
import torch.nn as nn
import torch

# The Gram loss uses relu1_1, relu2_1 and relu3_1, where texture lives; relu4_1 is mostly structure
GRAM_LAYERS = 3

def gram_matrix(features):
    """Channel-by-channel correlations averaged over positions: which features occur together, i.e. texture."""
    batch, channels, height, width = features.shape
    flat = features.flatten(2)
    # Averaging over positions only (not also channels) keeps the loss large enough for a normal-sized weight
    return flat @ flat.transpose(1, 2) / (height * width)

class CombinedObjective(nn.Module):
    LOSS_NAMES = ("content", "style", "style_gram", "identity", "identity_features", "tv")

    def __init__(
        self,
        encoder: VGGEncoder,
        lambda_content=1.0,
        lambda_style=10.0,
        lambda_style_gram=0.0,
        lambda_identity=1.0,
        lambda_identity_features=0.0,
        lambda_tv=1.0,
    ):
        super(CombinedObjective, self).__init__()

        self.register_buffer('lambdas', torch.tensor([lambda_content, lambda_style, lambda_style_gram, lambda_identity, lambda_identity_features, lambda_tv]))
        self.use_style_gram = lambda_style_gram > 0
        self.use_identity_features = lambda_identity_features > 0

        # Frozen VGG that measures the losses; the AdaIN model shares its own encoder here
        self.encoder = encoder
        self.style_loss = StyleLoss()
        self.identity_loss = IdentityLoss()
        self.total_variation_loss = TotalVariationLoss()

    def forward(self, stylized_images, target_features, content_images, style_images, identity_content, identity_style):
        """
        Returns the weighted total loss and the unweighted losses, ordered as LOSS_NAMES.
        target_features is the AdaIN output for AdaIN models, or None for models without one.
        """
        stylized_features = self.encoder(stylized_images)
        with torch.no_grad():
            content_features = self.encoder(content_images)
            style_features = self.encoder(style_images)

        if self.use_identity_features:
            identity_content_features = self.encoder(identity_content)
            identity_style_features = self.encoder(identity_style)

        zero = torch.zeros((), device=stylized_images.device)

        # Losses are computed in float32 even under autocast: bfloat16's 7 mantissa bits are too coarse for means, stds and pixel differences
        with torch.autocast(device_type=stylized_images.device.type, enabled=False):
            if target_features is not None:
                # AdaIN: the content target is the AdaIN output, so the decoder learns to invert AdaIN
                content_loss = F.mse_loss(stylized_features[-1].float(), target_features.float())
            else:
                # No AdaIN target (SANet): compare normalized features, so structure is kept while statistics change
                content_loss = F.mse_loss(
                    mean_variance_norm(stylized_features[-1].float()),
                    mean_variance_norm(content_features[-1].float()),
                )

            style_loss = sum(
                self.style_loss(stylized.float(), style.float()) for stylized, style in zip(stylized_features, style_features)
            )

            if self.use_style_gram:
                style_gram_loss = sum(
                    F.mse_loss(gram_matrix(stylized.float()), gram_matrix(style.float()))
                    for stylized, style in zip(stylized_features[:GRAM_LAYERS], style_features[:GRAM_LAYERS])
                )
            else:
                style_gram_loss = zero

            identity_loss = (
                self.identity_loss(identity_content.float(), content_images.float())
                + self.identity_loss(identity_style.float(), style_images.float())
            )

            if self.use_identity_features:
                identity_features_loss = sum(
                    F.mse_loss(identity.float(), target.float())
                    for identity, target in zip(identity_content_features + identity_style_features, content_features + style_features)
                )
            else:
                identity_features_loss = zero

            tv_loss = self.total_variation_loss(stylized_images.float())

            losses = torch.stack([content_loss, style_loss, style_gram_loss, identity_loss, identity_features_loss, tv_loss])
            total_loss = (self.lambdas * losses).sum()

        return total_loss, losses
