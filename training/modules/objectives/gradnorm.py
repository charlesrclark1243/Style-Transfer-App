import torch.nn as nn
import torch

class GradNorm(nn.Module):
    """
    GradNorm (Chen et al., 2018): learns loss weights so that each loss's gradient norm on a shared
    parameter tracks how quickly that loss is training relative to the others.
    """
    def __init__(self, num_losses: int, alpha: float = 1.5):
        super(GradNorm, self).__init__()

        self.num_losses = num_losses
        self.alpha = alpha

        self.weights = nn.Parameter(torch.ones(num_losses))
        self.initial_losses = None

    def update_weight_gradients(self, losses: torch.Tensor, shared_parameter: torch.Tensor):
        """
        Sets self.weights.grad from the GradNorm loss. The graph behind losses is retained,
        so the model's backward pass can run afterwards.
        """
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()

        # grad(w * L) = w * grad(L), so the unweighted norms are enough and no double backward is needed
        grad_norms = torch.stack([
            torch.autograd.grad(loss, shared_parameter, retain_graph=True)[0].norm()
            for loss in losses
        ])
        weighted_grad_norms = self.weights * grad_norms

        loss_ratios = losses.detach() / self.initial_losses
        inverse_training_rates = loss_ratios / loss_ratios.mean()
        target_grad_norms = (weighted_grad_norms.mean() * inverse_training_rates ** self.alpha).detach()

        gradnorm_loss = torch.abs(weighted_grad_norms - target_grad_norms).sum()
        self.weights.grad = torch.autograd.grad(gradnorm_loss, self.weights)[0]

    @torch.no_grad()
    def renormalize(self):
        # Keep weights positive and summing to num_losses, so only their ratios change
        self.weights.clamp_(min=1e-3)
        self.weights.mul_(self.num_losses / self.weights.sum())
