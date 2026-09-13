from utils.style_transfer_dataset import StyleTransferDataset
from utils.loops import main_loop

from modules.model import StyleTransferModel
from modules.objectives.combined_objective import CombinedObjective

from torchvision.transforms import CenterCrop, Compose, RandomCrop, Resize, ToTensor
import torch.optim as optim
import torch

from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a style transfer model.")

    parser.add_argument('--content-dir', type=str, required=True, help='Path to the content images directory.')
    parser.add_argument('--style-dir', type=str, required=True, help='Path to the style images directory.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--gradnorm-alpha', type=float, default=1.5, help='GradNorm asymmetry; higher values balance training rates more strongly.')
    parser.add_argument('--gradnorm-learning-rate', type=float, default=0.01, help='Learning rate for the GradNorm loss weights.')

    return parser.parse_args()

def get_dataloaders(
    content_dir: str,
    style_dir: str,
    batch_size: int = 8
):
    # Resize the shorter side then crop, so non-square images aren't stretched
    train_transform = Compose([
        Resize(512),
        RandomCrop(512),
        ToTensor(),
    ])
    val_transform = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
    ])

    content_dir = Path(content_dir)
    style_dir = Path(style_dir)

    train_dataset = StyleTransferDataset(content_dir / "train", style_dir / "train", transform=train_transform)
    val_dataset = StyleTransferDataset(content_dir / "val", style_dir / "val", transform=val_transform, random_style=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def main():
    args = parse_args()

    train_loader, val_loader = get_dataloaders(args.content_dir, args.style_dir, batch_size=args.batch_size)

    model = StyleTransferModel()
    objective = CombinedObjective(gradnorm_alpha=args.gradnorm_alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    weight_optimizer = optim.Adam(objective.gradnorm.parameters(), lr=args.gradnorm_learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        weight_optimizer,
        objective,
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.num_epochs,
    )

if __name__ == "__main__":
    main()
