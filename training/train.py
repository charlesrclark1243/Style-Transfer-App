import argparse
import os
from pathlib import Path
from typing import Any

import torch
from modules.model import StyleTransferModel, read_checkpoint
from modules.objectives.combined_objective import CombinedObjective
from torch import optim
from torchvision.transforms import CenterCrop, Compose, RandomCrop, Resize, ToTensor
from utils.loops import main_loop
from utils.style_transfer_dataset import StyleTransferDataset


def positive_int(value: Any) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(value: Any) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a style transfer model.")

    parser.add_argument(
        "--content-dir",
        type=str,
        required=True,
        help="Path to the content images directory.",
    )
    parser.add_argument(
        "--style-dir",
        type=str,
        required=True,
        help="Path to the style images directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to continue training from. Epoch numbering continues; the optimizer starts fresh.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for (in addition to any resumed epochs).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer. 1e-3 makes activations explode within a few steps.",
    )
    parser.add_argument(
        "--lambda-content",
        type=float,
        default=1.0,
        help="Weight of the VGG content loss.",
    )
    parser.add_argument(
        "--lambda-style",
        type=float,
        default=10.0,
        help="Weight of the VGG style loss. Higher values stylize more strongly.",
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=1.0,
        help="Weight of the identity reconstruction loss.",
    )
    parser.add_argument(
        "--lambda-tv",
        type=float,
        default=1.0,
        help="Weight of the total variation smoothness loss.",
    )
    parser.add_argument(
        "--max-train-images",
        type=positive_int,
        default=None,
        help="Maximum number of content and style training images, sampled at random.",
    )
    parser.add_argument(
        "--max-val-images",
        type=positive_int,
        default=None,
        help="Maximum number of content and style validation images, sampled at random.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed for which images --max-train-images and --max-val-images sample, so runs can use the same subset.",
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=min(8, os.cpu_count() or 1),
        help="Background processes that load images. 0 loads them in the training process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="bf16 runs forward passes in bfloat16 mixed precision, using about 40%% less GPU memory than fp32.",
    )

    return parser.parse_args()


def get_dataloaders(
    content_dir: str,
    style_dir: str,
    batch_size: int = 8,
    max_train_images: int | None = None,
    max_val_images: int | None = None,
    subset_seed: int = 0,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Resize the shorter side then crop, so non-square images aren't stretched
    train_transform = Compose(
        [
            Resize(512),
            RandomCrop(256),
            ToTensor(),
        ]
    )
    val_transform = Compose(
        [
            Resize(512),
            CenterCrop(256),
            ToTensor(),
        ]
    )

    content_dir = Path(content_dir)
    style_dir = Path(style_dir)

    train_dataset = StyleTransferDataset(
        content_dir / "train",
        style_dir / "train",
        transform=train_transform,
        max_images=max_train_images,
        subset_seed=subset_seed,
    )
    val_dataset = StyleTransferDataset(
        content_dir / "val",
        style_dir / "val",
        transform=val_transform,
        random_style=False,
        max_images=max_val_images,
        subset_seed=subset_seed,
    )

    # Persistent workers stay alive between epochs instead of being restarted each time
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_options
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    model = StyleTransferModel()
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(read_checkpoint(checkpoint))
        start_epoch = checkpoint["epoch"]
        print(
            f"Resumed from {args.resume} at epoch {start_epoch}; the optimizer starts fresh"
        )

    train_loader, val_loader = get_dataloaders(
        args.content_dir,
        args.style_dir,
        batch_size=args.batch_size,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        subset_seed=args.subset_seed,
        num_workers=args.num_workers,
    )

    objective = CombinedObjective(
        model.encoder,
        lambda_content=args.lambda_content,
        lambda_style=args.lambda_style,
        lambda_identity=args.lambda_identity,
        lambda_tv=args.lambda_tv,
    )
    # Only the decoder trains; the VGG encoder is frozen
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixed_precision = args.precision == "bf16"
    if mixed_precision and not (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    ):
        print("bfloat16 is not supported on this device; falling back to fp32")
        mixed_precision = False

    main_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        objective,
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.num_epochs,
        mixed_precision=mixed_precision,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
