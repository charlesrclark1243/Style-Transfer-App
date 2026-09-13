from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
import random

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None, random_style=True):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.transform = transform
        self.random_style = random_style

        self.content_images = sorted(path for path in self.content_dir.glob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)
        self.style_images = sorted(path for path in self.style_dir.glob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)

        if not self.content_images:
            raise ValueError(f"No content images found in {content_dir}")
        if not self.style_images:
            raise ValueError(f"No style images found in {style_dir}")

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        content_path = self.content_images[idx]
        # Fixed pairing keeps validation loss comparable across epochs
        if self.random_style:
            style_path = random.choice(self.style_images)
        else:
            style_path = self.style_images[idx % len(self.style_images)]

        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)
        
        return content_image, style_image