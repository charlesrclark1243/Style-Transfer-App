import torch.nn.functional as F
import torch.nn as nn
import torch

def mean_variance_norm(x, eps=1e-5):
    """Normalizes each channel to zero mean and unit std over its spatial positions."""
    return (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + eps)

class StyleAttention(nn.Module):
    """
    Cross-attention from content to style features (SANet, Park & Lee 2019). Each content position takes a mix of
    style features, weighted by how similar the normalized content and style features are.

    With residual=True the content features are added back, as in SANet. With residual=False only the attended style
    features are returned, and the output projection starts at zero, so the block begins as a no-op.
    """
    def __init__(self, channels: int, num_heads: int = 1, residual: bool = True):
        super(StyleAttention, self).__init__()

        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.residual = residual

        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

        if not residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, content_features, style_features):
        # Runs in float32 even under autocast; bfloat16 is too coarse for the normalization and attention weights
        with torch.autocast(device_type=content_features.device.type, enabled=False):
            content_features = content_features.float()
            style_features = style_features.float()

            batch, channels, height, width = content_features.shape
            head_dim = channels // self.num_heads

            def to_heads(x):
                # B × C × H × W  ->  B × heads × positions × head_dim. Contiguous because the memory-efficient kernel
                # needs stride 1 in the last dimension; otherwise SDPA silently builds the full attention matrix
                return x.flatten(2).transpose(1, 2).reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2).contiguous()

            # Normalizing before query and key makes matching depend on structure rather than each image's statistics
            query = to_heads(self.query(mean_variance_norm(content_features)))
            key = to_heads(self.key(mean_variance_norm(style_features)))
            value = to_heads(self.value(style_features))

            # Memory-efficient kernels avoid materializing the full positions × positions attention matrix
            attended = F.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(batch, height * width, channels).transpose(1, 2).reshape(batch, channels, height, width)

            output = self.out(attended)
            return content_features + output if self.residual else output
