import torch.nn.functional as F
import torch.nn as nn
import torch

from torchvision.models import vgg19, VGG19_Weights

# Slice ends in vgg19().features for relu1_1, relu2_1, relu3_1 and relu4_1
VGG_LAYER_ENDS = [2, 7, 12, 21]

class VGGPerceptualContentLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualContentLoss, self).__init__()
        vgg = list(vgg19(weights=VGG19_Weights.DEFAULT).features.children())
        starts = [0] + VGG_LAYER_ENDS[:-1]
        self.vgg_blocks = nn.ModuleList(
            nn.Sequential(*vgg[start:end]) for start, end in zip(starts, VGG_LAYER_ENDS)
        ).eval()
        for param in self.vgg_blocks.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract_features(self, image):
        """Returns a list of features at relu1_1, relu2_1, relu3_1 and relu4_1."""
        x = (image - self.mean) / self.std

        features = []
        for block in self.vgg_blocks:
            x = block(x)
            features.append(x)

        return features

    def forward(self, stylized_image, content_image):
        # Content is compared at relu4_1 only; the shallower layers are for the style loss
        stylized_features = self.extract_features(stylized_image)[-1]
        content_features = self.extract_features(content_image)[-1]

        loss = F.mse_loss(stylized_features, content_features)
        return loss
