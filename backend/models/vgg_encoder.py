import torch
from torch import nn
from torchvision.models import vgg19

# Slice ends in vgg19().features for relu1_1, relu2_1, relu3_1 and relu4_1
VGG_LAYER_ENDS = [2, 7, 12, 21]


class VGGEncoder(nn.Module):
    """
    Frozen ImageNet VGG19 up to relu4_1. Takes images in [0, 1] and returns a list of features
    at relu1_1, relu2_1, relu3_1 and relu4_1.
    """

    def __init__(self):
        super().__init__()

        # No pretrained download: the trained checkpoint already contains these VGG weights
        vgg = list(vgg19(weights=None).features.children())
        starts = [0] + VGG_LAYER_ENDS[:-1]
        self.blocks = nn.ModuleList(
            nn.Sequential(*vgg[start:end]) for start, end in zip(starts, VGG_LAYER_ENDS)
        )
        for param in self.blocks.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = (image - self.mean) / self.std

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features
