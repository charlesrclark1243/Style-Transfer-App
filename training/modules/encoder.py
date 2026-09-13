import torch.nn as nn

def split_before_last_pool(layers: nn.Sequential):
    """Splits a head at its last max pool: the first part outputs features at 1/4 resolution, the rest takes them to 1/8."""
    last_pool = max(i for i, layer in enumerate(layers) if isinstance(layer, nn.MaxPool2d))
    return layers[:last_pool], layers[last_pool:]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.shared_encoder = SharedEncoder()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()

    def forward(self, content_x, style_x, return_quarter_features=False):
        content_shared_features = self.shared_encoder(content_x)
        style_shared_features = self.shared_encoder(style_x)

        if return_quarter_features:
            content_features, content_quarter_features = self.content_encoder(content_shared_features, return_quarter_features=True)
            style_features, style_quarter_features = self.style_encoder(style_shared_features, return_quarter_features=True)

            return content_features, style_features, content_quarter_features, style_quarter_features

        content_features = self.content_encoder(content_shared_features)
        style_features = self.style_encoder(style_shared_features)

        return content_features, style_features

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)

        return x

class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, return_quarter_features=False):
        if return_quarter_features:
            to_quarter, to_eighth = split_before_last_pool(self.layers)
            quarter_features = to_quarter(x)

            return to_eighth(quarter_features), quarter_features

        x = self.layers(x)

        return x

class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        self.spatial_embedding = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.global_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x, return_quarter_features=False):
        if return_quarter_features:
            to_quarter, to_eighth = split_before_last_pool(self.spatial_embedding)
            quarter_features = to_quarter(x)
            spatial_features = to_eighth(quarter_features)
        else:
            spatial_features = self.spatial_embedding(x)

        global_features = self.global_embedding(spatial_features)
        style_features = spatial_features * (1 + global_features)

        return (style_features, quarter_features) if return_quarter_features else style_features
