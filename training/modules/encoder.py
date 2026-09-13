import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.shared_encoder = SharedEncoder()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()

    def forward(self, content_x, style_x):
        content_shared_features = self.shared_encoder(content_x)
        style_shared_features = self.shared_encoder(style_x)

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

    def forward(self, x):
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

    def forward(self, x):
        spatial_features = self.spatial_embedding(x)
        global_features = self.global_embedding(spatial_features)

        return spatial_features * (1 + global_features)