import torch.nn.functional as F
import torch.nn as nn

class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, generated_image, target_image):
        identity_loss = F.l1_loss(generated_image, target_image)
        return identity_loss