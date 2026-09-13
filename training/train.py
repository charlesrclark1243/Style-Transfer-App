from utils.style_transfer_dataset import StyleTransferDataset
from utils.loops import main_loop

from modules.model import MODELS, read_checkpoint
from modules.vgg_encoder import VGGEncoder
from modules.objectives.combined_objective import CombinedObjective

from torchvision.transforms import CenterCrop, Compose, RandomCrop, Resize, ToTensor
import torch.optim as optim
import torch

from pathlib import Path
import argparse
import sys
import os
import re

# Roughly SANet's weights for the scratch models; the identity losses anchor a from-scratch encoder to reconstructing its inputs
SCRATCH_LOSS_WEIGHTS = dict(lambda_content=1.0, lambda_style=3.0, lambda_style_gram=0.0, lambda_identity=1.0, lambda_identity_features=50.0, lambda_tv=1.0)

LOSS_WEIGHT_DEFAULTS = {
    'vgg-adain': dict(lambda_content=1.0, lambda_style=10.0, lambda_style_gram=0.0, lambda_identity=1.0, lambda_identity_features=0.0, lambda_tv=1.0),
    'scratch-attention': SCRATCH_LOSS_WEIGHTS,
    'scratch-attention-multiscale': SCRATCH_LOSS_WEIGHTS,
    'scratch-attention-styled': SCRATCH_LOSS_WEIGHTS,
}

def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value

def non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Train a style transfer model.")

    parser.add_argument('--content-dir', type=str, required=True, help='Path to the content images directory.')
    parser.add_argument('--style-dir', type=str, required=True, help='Path to the style images directory.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--model', type=str, choices=list(MODELS), default=None, help='vgg-adain (default): frozen VGG encoder with AdaIN. scratch-attention: the from-scratch two-headed encoder with cross-attention. scratch-attention-multiscale: adds a second attention block at 1/4 resolution. scratch-attention-styled: multiscale plus style modulation in every decoder block. Taken from the checkpoint when resuming, unless given.')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to continue training from. It can also seed a model that adds modules to it (e.g. scratch-attention-multiscale into scratch-attention-styled). Epoch numbering continues; the optimizer starts fresh.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train for (in addition to any resumed epochs).')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer. 1e-3 makes activations explode within a few steps.')
    parser.add_argument('--lambda-content', type=float, default=None, help='Weight of the VGG content loss. Default depends on --model.')
    parser.add_argument('--lambda-style', type=float, default=None, help='Weight of the VGG mean/std style loss. Higher values stylize more strongly. Default depends on --model.')
    parser.add_argument('--lambda-style-gram', type=float, default=None, help='Weight of the Gram-matrix style loss at relu1_1-relu3_1, which matches texture. Default 0 (off).')
    parser.add_argument('--lambda-identity', type=float, default=None, help='Weight of the pixel identity reconstruction loss. Default depends on --model.')
    parser.add_argument('--lambda-identity-features', type=float, default=None, help='Weight of the VGG feature identity reconstruction loss. Default depends on --model.')
    parser.add_argument('--lambda-tv', type=float, default=None, help='Weight of the total variation smoothness loss. Default depends on --model.')
    parser.add_argument('--max-train-images', type=positive_int, default=None, help='Maximum number of content and style training images, sampled at random.')
    parser.add_argument('--max-val-images', type=positive_int, default=None, help='Maximum number of content and style validation images, sampled at random.')
    parser.add_argument('--subset-seed', type=int, default=0, help='Seed for which images --max-train-images and --max-val-images sample, so runs can use the same subset.')
    parser.add_argument('--num-workers', type=non_negative_int, default=min(8, os.cpu_count() or 1), help='Background processes that load images. 0 loads them in the training process.')
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp32'], default='bf16', help='bf16 runs forward passes in bfloat16 mixed precision, using about 40%% less GPU memory than fp32.')

    return parser.parse_args()

def get_dataloaders(
    content_dir: str,
    style_dir: str,
    batch_size: int = 8,
    max_train_images: int = None,
    max_val_images: int = None,
    subset_seed: int = 0,
    num_workers: int = 0,
):
    # Resize the shorter side then crop, so non-square images aren't stretched
    train_transform = Compose([
        Resize(512),
        RandomCrop(256),
        ToTensor(),
    ])
    val_transform = Compose([
        Resize(512),
        CenterCrop(256),
        ToTensor(),
    ])

    content_dir = Path(content_dir)
    style_dir = Path(style_dir)

    train_dataset = StyleTransferDataset(content_dir / "train", style_dir / "train", transform=train_transform, max_images=max_train_images, subset_seed=subset_seed)
    val_dataset = StyleTransferDataset(content_dir / "val", style_dir / "val", transform=val_transform, random_style=False, max_images=max_val_images, subset_seed=subset_seed)

    # Persistent workers stay alive between epochs instead of being restarted each time
    loader_options = dict(batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader

def checkpoint_epoch(checkpoint: dict, checkpoint_path: str):
    # Older checkpoints don't record their epoch, so fall back to the epoch_N filename
    if "epoch" in checkpoint:
        return checkpoint["epoch"]

    match = re.fullmatch(r"epoch_(\d+)", Path(checkpoint_path).stem)
    return int(match.group(1)) if match else 0

def main():
    args = parse_args()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_model_name, state_dict = read_checkpoint(checkpoint)
        model_name = args.model or checkpoint_model_name
        start_epoch = checkpoint_epoch(checkpoint, args.resume)
    else:
        model_name = args.model or 'vgg-adain'
        start_epoch = 0

    # Built before the data loaders, so a checkpoint that doesn't fit fails immediately
    model = MODELS[model_name]()
    if args.resume:
        # Non-strict so a checkpoint can seed a model that adds modules; anything else that doesn't line up is an error
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as error:
            # Shape mismatches raise even when loading non-strictly
            sys.exit(f"{args.resume} is a {checkpoint_model_name} checkpoint and doesn't fit {model_name}:\n{error}")
        if unexpected or (missing and model_name == checkpoint_model_name):
            sys.exit(
                f"{args.resume} is a {checkpoint_model_name} checkpoint and doesn't fit {model_name} "
                f"(missing: {missing[:3]}, unexpected: {unexpected[:3]})"
            )

        new_modules = sorted({key.split('.')[0] for key in missing})
        added = f"; newly initialized: {', '.join(new_modules)}" if new_modules else ""
        print(f"Resumed {model_name} from {args.resume} ({checkpoint_model_name}) at epoch {start_epoch}; the optimizer starts fresh{added}")

    train_loader, val_loader = get_dataloaders(
        args.content_dir,
        args.style_dir,
        batch_size=args.batch_size,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        subset_seed=args.subset_seed,
        num_workers=args.num_workers,
    )

    loss_weights = {
        name: default if getattr(args, name) is None else getattr(args, name)
        for name, default in LOSS_WEIGHT_DEFAULTS[model_name].items()
    }
    print(f"Model: {model_name} | loss weights: {loss_weights}")

    # VGG measures the losses; the AdaIN model already holds a frozen copy, so it is shared
    loss_encoder = model.encoder if isinstance(model.encoder, VGGEncoder) else VGGEncoder()
    objective = CombinedObjective(loss_encoder, **loss_weights)

    # Only trainable parameters; a VGG encoder stays frozen
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixed_precision = args.precision == 'bf16'
    if mixed_precision and not (device.type == 'cuda' and torch.cuda.is_bf16_supported()):
        print("bfloat16 is not supported on this device; falling back to fp32")
        mixed_precision = False

    main_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        objective,
        output_dir=args.output_dir,
        model_name=model_name,
        device=device,
        num_epochs=args.num_epochs,
        mixed_precision=mixed_precision,
        start_epoch=start_epoch,
    )

if __name__ == "__main__":
    main()
