from modules.model import StyleTransferModel
from modules.objectives.combined_objective import CombinedObjective

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from pathlib import Path
from tqdm import tqdm

def format_losses(total_loss: float, losses: torch.Tensor) -> str:
    terms = ", ".join(f"{name}={value:.3g}" for name, value in zip(CombinedObjective.LOSS_NAMES, losses.tolist()))
    return f"Loss: {total_loss:.4f} | {terms}"

def train_loop(
    model: StyleTransferModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    objective: CombinedObjective,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    epoch: int = 0,
    num_epochs: int = 10,
    mixed_precision: bool = False,
) -> float:

    model.to(device)
    model.train()

    total_loss = 0.0
    loss_sums = torch.zeros(len(CombinedObjective.LOSS_NAMES), device=device)

    loop = tqdm(train_loader)
    for batch_idx, (content_images, style_images) in enumerate(loop):
        loop.set_description_str(f"Train: Batch {batch_idx + 1}/{len(train_loader)} [Epoch {epoch + 1}/{num_epochs}]")

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        optimizer.zero_grad()

        # Forward passes run in bfloat16 when enabled; the backward pass stays outside autocast
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
            stylized_images, target_features = model.transfer(content_images, style_images)

            identity_content = model(content_images, content_images)
            identity_style = model(style_images, style_images)

            loss, losses = objective(stylized_images, target_features, content_images, style_images, identity_content, identity_style)

        loss.backward()
        optimizer.step()

        # Running averages, so the bar shows the epoch mean of each loss rather than one noisy batch
        total_loss += loss.item()
        loss_sums += losses.detach()
        loop.set_postfix_str(format_losses(total_loss / (batch_idx + 1), loss_sums / (batch_idx + 1)))

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
    mixed_precision: bool = False,
) -> float:

    model.to(device)
    model.eval()

    total_loss = 0.0
    loss_sums = torch.zeros(len(CombinedObjective.LOSS_NAMES), device=device)

    loop = tqdm(val_loader)
    with torch.no_grad():
        for batch_idx, (content_images, style_images) in enumerate(loop):
            loop.set_description_str(f"Val: Batch {batch_idx + 1}/{len(val_loader)} [Epoch {epoch + 1}/{num_epochs}]")

            content_images = content_images.to(device)
            style_images = style_images.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
                stylized_images, target_features = model.transfer(content_images, style_images)

                identity_content = model(content_images, content_images)
                identity_style = model(style_images, style_images)

                loss, losses = objective(stylized_images, target_features, content_images, style_images, identity_content, identity_style)

            total_loss += loss.item()
            loss_sums += losses
            loop.set_postfix_str(format_losses(total_loss / (batch_idx + 1), loss_sums / (batch_idx + 1)))

    return total_loss / len(val_loader)

def main_loop(
    model: StyleTransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    objective: CombinedObjective,
    output_dir: Path,
    model_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs: int = 10,
    mixed_precision: bool = False,
    start_epoch: int = 0,
):
    # Create the output directory up front so a bad path fails before training, not after
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objective.to(device)

    # When resuming, epochs continue from start_epoch so checkpoint numbering carries on
    final_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, final_epoch):
        _ = train_loop(model, train_loader, optimizer, objective, device, epoch, final_epoch, mixed_precision=mixed_precision)
        _ = val_loop(model, val_loader, objective, device, epoch, final_epoch, mixed_precision=mixed_precision)

        # The model name lets eval.py rebuild the right architecture
        torch.save({"model": model_name, "epoch": epoch + 1, "state_dict": model.state_dict()}, output_dir / f"epoch_{epoch + 1}.pth")
