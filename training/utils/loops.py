from modules.model import StyleTransferModel
from modules.objectives.combined_objective import CombinedObjective

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from pathlib import Path
from tqdm import tqdm

def train_loop(
    model: StyleTransferModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    weight_optimizer: optim.Optimizer,
    objective: CombinedObjective,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    epoch: int = 0,
    num_epochs: int = 10,
) -> float:

    model.to(device)
    model.train()

    total_loss = 0.0

    loop = tqdm(train_loader)
    for batch_idx, (content_images, style_images) in enumerate(loop):
        loop.set_description_str(f"Train: Batch {batch_idx + 1}/{len(train_loader)} [Epoch {epoch + 1}/{num_epochs}]")

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        optimizer.zero_grad()
        stylized_images = model(content_images, style_images)

        identity_content = model(content_images, content_images)
        identity_style = model(style_images, style_images)

        balanced_losses, tv_loss = objective(stylized_images, content_images, style_images, identity_content, identity_style)

        # GradNorm needs the graph intact, so it runs before the model's backward pass
        objective.gradnorm.update_weight_gradients(balanced_losses, model.last_shared_parameter)

        loss = objective.weighted_total(balanced_losses, tv_loss)
        loss.backward()
        optimizer.step()

        weight_optimizer.step()
        objective.gradnorm.renormalize()

        total_loss += loss.item()
        weights = ", ".join(
            f"{name}={weight:.2f}" for name, weight in zip(objective.BALANCED_LOSSES, objective.gradnorm.weights.tolist())
        )
        loop.set_postfix_str(f"Combined Loss: {loss.item():.4f} | Weights: {weights}")

    return total_loss / len(train_loader)

def val_loop(
    model: StyleTransferModel,
    val_loader: DataLoader,
    objective: CombinedObjective,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    epoch: int = 0,
    num_epochs: int = 10,
) -> float:

    model.to(device)
    model.eval()

    total_loss = 0.0

    loop = tqdm(val_loader)
    with torch.no_grad():
        for batch_idx, (content_images, style_images) in enumerate(loop):
            loop.set_description_str(f"Val: Batch {batch_idx + 1}/{len(val_loader)} [Epoch {epoch + 1}/{num_epochs}]")

            content_images = content_images.to(device)
            style_images = style_images.to(device)

            stylized_images = model(content_images, style_images)

            identity_content = model(content_images, content_images)
            identity_style = model(style_images, style_images)

            balanced_losses, tv_loss = objective(stylized_images, content_images, style_images, identity_content, identity_style)
            # Fixed lambdas only, so validation loss stays comparable across epochs as the GradNorm weights move
            loss = objective.weighted_total(balanced_losses, tv_loss, use_gradnorm_weights=False)
            total_loss += loss.item()

            loop.set_postfix_str(f"Combined Loss: {loss.item():.4f}")

    return total_loss / len(val_loader)

def main_loop(
    model: StyleTransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    weight_optimizer: optim.Optimizer,
    objective: CombinedObjective,
    output_dir: Path,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs: int = 10
):
    # Create the output directory up front so a bad path fails before training, not after
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objective.to(device)

    for epoch in range(num_epochs):
        _ = train_loop(model, train_loader, optimizer, weight_optimizer, objective, device, epoch, num_epochs)
        _ = val_loop(model, val_loader, objective, device, epoch, num_epochs)

        torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1}.pth")
