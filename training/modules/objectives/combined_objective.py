from modules.objectives.vgg_perceptual_content_loss import VGGPerceptualContentLoss
from modules.objectives.style_loss import StyleLoss
from modules.objectives.identity_loss import IdentityLoss
from modules.objectives.total_variation_loss import TotalVariationLoss
from modules.objectives.gradnorm import GradNorm

import torch.nn.functional as F
import torch.nn as nn
import torch

class CombinedObjective(nn.Module):
    BALANCED_LOSSES = ("content", "style", "identity")

    def __init__(self, lambda_content=1.0, lambda_style=1.0, lambda_identity=1.0, lambda_tv=1.0, gradnorm_alpha=1.5):
        super(CombinedObjective, self).__init__()

        # The lambdas scale the GradNorm-balanced losses, so they act as preferences between objectives
        self.register_buffer('lambdas', torch.tensor([lambda_content, lambda_style, lambda_identity]))
        # Total variation is a regularizer rather than an objective, so it keeps a fixed weight outside GradNorm
        self.lambda_tv = lambda_tv

        self.vgg = VGGPerceptualContentLoss()
        self.style_loss = StyleLoss()
        self.identity_loss = IdentityLoss()
        self.total_variation_loss = TotalVariationLoss()

        self.gradnorm = GradNorm(num_losses=len(self.BALANCED_LOSSES), alpha=gradnorm_alpha)

    def forward(self, stylized_images, content_images, style_images, identity_content, identity_style):
        """Returns the unweighted losses: a tensor ordered as BALANCED_LOSSES, and the total variation loss."""
        # Extract stylized features once and reuse them for both content and style losses
        stylized_features = self.vgg.extract_features(stylized_images)
        with torch.no_grad():
            content_features = self.vgg.extract_features(content_images)
            style_features = self.vgg.extract_features(style_images)

        content_loss = F.mse_loss(stylized_features[-1], content_features[-1])
        style_loss = sum(
            self.style_loss(stylized, style) for stylized, style in zip(stylized_features, style_features)
        )
        identity_loss = self.identity_loss(identity_content, content_images) + self.identity_loss(identity_style, style_images)
        tv_loss = self.total_variation_loss(stylized_images)

        return torch.stack([content_loss, style_loss, identity_loss]), tv_loss

    def weighted_total(self, balanced_losses, tv_loss, use_gradnorm_weights=True):
        # GradNorm weights are detached so the model's backward pass leaves their gradient to GradNorm
        weights = self.lambdas * self.gradnorm.weights.detach() if use_gradnorm_weights else self.lambdas
        return (weights * balanced_losses).sum() + self.lambda_tv * tv_loss
