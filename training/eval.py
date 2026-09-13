from modules.model import MODELS, read_checkpoint

from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from torchvision.transforms.functional import center_crop
import torch

import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a style transfer model.")

    parser.add_argument('--content-image', type=str, required=True, help='Path to the content image.')
    parser.add_argument('--style-image', type=str, required=True, help='Path to the style image.')
    parser.add_argument('--model-checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--size', type=int, default=512, help='Resize the shorter side of both images to this, matching training.')
    parser.add_argument('--output', type=str, default=None, help='Path to save the stylized image. Shows it in a window if omitted.')

    args = parser.parse_args()
    if args.size < 8:
        parser.error(f"--size must be at least 8, got {args.size}")

    return args

def load_model(checkpoint_path: str, device: torch.device):
    model_name, state_dict = read_checkpoint(torch.load(checkpoint_path, map_location=device))

    model = MODELS[model_name]()
    model.load_state_dict(state_dict)

    return model.to(device)

def crop_to_multiple_of_8(image: torch.Tensor):
    # The encoder halves the size three times, so other sizes would come back a few pixels smaller
    height, width = image.shape[-2:]
    return center_crop(image, [height - height % 8, width - width % 8])

def infer(
    model: torch.nn.Module,
    content_image_path: str,
    style_image_path: str,
    size: int = 512,
    device: torch.device = torch.device("cpu")
):
    model.eval()

    transform = Compose([
        Resize(size),
        ToTensor(),
        Lambda(crop_to_multiple_of_8),
    ])

    content_image = transform(Image.open(content_image_path).convert('RGB')).unsqueeze(0).to(device)
    style_image = transform(Image.open(style_image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        stylized_image = model(content_image, style_image)

    # The decoder's output is unbounded, so clip to a valid image range
    return stylized_image.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

def show_stylized_image(stylized_image):
    plt.imshow(stylized_image)
    plt.axis('off')
    plt.show()

def save_stylized_image(stylized_image, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray((stylized_image * 255).round().astype('uint8')).save(output_path)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_checkpoint, device)
    stylized_image = infer(model, args.content_image, args.style_image, size=args.size, device=device)

    if args.output:
        save_stylized_image(stylized_image, args.output)
    else:
        show_stylized_image(stylized_image)

if __name__ == "__main__":
    main()
