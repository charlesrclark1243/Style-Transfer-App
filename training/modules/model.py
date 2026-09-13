from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.adain import AdaIN

import torch.nn as nn

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.adain = AdaIN()

    @property
    def last_shared_parameter(self):
        # Final decoder conv: every loss depends on it, so GradNorm measures gradient norms here
        return self.decoder.layers[-2].weight

    def forward(self, content_image, style_image):
        content_features, style_features = self.encoder(content_image, style_image)
        stylized_features = self.adain(content_features, style_features)
        stylized_image = self.decoder(stylized_features)

        return stylized_image