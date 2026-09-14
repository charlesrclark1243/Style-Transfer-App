import torch
import torch.nn.functional as F
from modules.objectives.identity_loss import IdentityLoss
from modules.objectives.style_loss import StyleLoss
from modules.objectives.total_variation_loss import TotalVariationLoss
from modules.vgg_encoder import VGGEncoder
from torch import nn


class CombinedObjective(nn.Module):
    LOSS_NAMES = ("content", "style", "identity", "tv")

    def __init__(
        self,
        encoder: VGGEncoder,
        lambda_content: float = 1.0,
        lambda_style: float = 10.0,
        lambda_identity: float = 1.0,
        lambda_tv: float = 1.0,
    ):
        super().__init__()

        self.register_buffer(
            "lambdas",
            torch.tensor([lambda_content, lambda_style, lambda_identity, lambda_tv]),
        )

        # Shares the model's frozen VGG, so losses are measured in the feature space AdaIN operates in
        self.encoder = encoder
        self.style_loss = StyleLoss()
        self.identity_loss = IdentityLoss()
        self.total_variation_loss = TotalVariationLoss()

    def forward(
        self,
        stylized_images: torch.Tensor,
        target_features: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        identity_content: torch.Tensor,
        identity_style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the weighted total loss and the unweighted losses, ordered as LOSS_NAMES."""
        stylized_features = self.encoder(stylized_images)
        with torch.no_grad():
            style_features = self.encoder(style_images)

        # Losses are computed in float32 even under autocast: bfloat16's 7 mantissa bits are too coarse for means, stds and pixel differences
        with torch.autocast(device_type=stylized_images.device.type, enabled=False):
            # As in AdaIN, the content target is the AdaIN output rather than the content image's features,
            # so the decoder learns to invert AdaIN
            content_loss = F.mse_loss(
                stylized_features[-1].float(), target_features.float()
            )
            style_loss = sum(
                self.style_loss(stylized.float(), style.float())
                for stylized, style in zip(stylized_features, style_features)
            )
            identity_loss = self.identity_loss(
                identity_content.float(), content_images.float()
            ) + self.identity_loss(identity_style.float(), style_images.float())
            tv_loss = self.total_variation_loss(stylized_images.float())

            losses = torch.stack([content_loss, style_loss, identity_loss, tv_loss])
            total_loss = (self.lambdas * losses).sum()

        return total_loss, losses
