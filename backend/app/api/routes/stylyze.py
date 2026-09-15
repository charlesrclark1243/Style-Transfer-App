import logging
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Annotated

import torch
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from models.model import StyleTransferModel, read_checkpoint
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import Resize, ToTensor

logger = logging.getLogger(__name__)

# Resolved from this file rather than the working directory, so the app finds the checkpoint however it's launched.
# MODEL_CHECKPOINT overrides it, e.g. for weights mounted into the container
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "epoch_6.pth"
CHECKPOINT_PATH = Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))


def load_model(checkpoint_path: Path) -> StyleTransferModel | None:
    """Loads the model, or returns None if the checkpoint is missing so the rest of the API still starts."""
    if not checkpoint_path.is_file():
        logger.warning(
            "No model checkpoint at %s; /api/stylyze will return 503", checkpoint_path
        )
        return None

    model = StyleTransferModel()
    model.load_state_dict(
        read_checkpoint(torch.load(checkpoint_path, map_location="cpu"))
    )

    return model.eval()


model = load_model(CHECKPOINT_PATH)

semaphore = threading.Semaphore(1)  # Limit to one inference at a time
torch.set_num_threads(4)

router = APIRouter()

ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_MEM_SIZE = 10 * 1024 * 1024  # 10 MB

MAX_PIXELS = 50 * 1_000_000  # 50 million pixels MAX (warning past 25 million)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS // 2  # Prevent decompression bomb DOS attacks


def load_image(file: UploadFile):
    """
    Load and validate an image file.

    Args:
        file (UploadFile): The image file to load.

    Returns:
        Image: The loaded and validated image.

    Raises:
        HTTPException: If the image is invalid or exceeds size limits.
    """

    # safeguards
    if file.content_type not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Invalid image format. Only JPEG and PNG are supported.",
        )

    if file.size is not None and file.size > MAX_MEM_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image size exceeds the maximum allowed size of {MAX_MEM_SIZE / (1024 * 1024)} MB.",
        )
    elif file.size is None:
        raise HTTPException(
            status_code=400,
            detail="Image size could not be determined. Please ensure the image is valid.",
        )

    # load and verify, then return if it passes
    try:
        image = Image.open(BytesIO(file.file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. The file could not be opened or is corrupted.",
        )

    return image


@router.post("/stylyze")
def stylyze(
    content: Annotated[UploadFile, File(...)],
    style: Annotated[UploadFile, File(...)],
    alpha: Annotated[float, Form(ge=0.0, le=1.0)] = 1.0,
):
    """
    Perform style transfer on the uploaded content and style images using the pre-trained model. Returns the stylized image.

    Args:
        content (UploadFile): The content image file.
        style (UploadFile): The style image file.
        alpha (float): Style strength, from 0 (keep the content's look) to 1 (full style). Defaults to 1.

    Returns:
        Response: The stylized image in JPEG format.
    """

    if model is None:
        raise HTTPException(
            status_code=503, detail="The style transfer model is not loaded."
        )

    content_image = Resize(512)(load_image(content))
    style_image = Resize(512)(load_image(style))

    width, height = content_image.size
    content_image = content_image.crop((0, 0, width - width % 8, height - height % 8))

    content_image = ToTensor()(content_image).unsqueeze(0)  # Add batch dimension
    style_image = ToTensor()(style_image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad(), semaphore:
        stylyzed_image = model(content_image, style_image, alpha)

    # Convert the output tensor to a PIL image
    stylyzed_image = stylyzed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stylyzed_image = Image.fromarray(
        (stylyzed_image.clip(0, 1) * 255).round().astype("uint8")
    )

    buffer = BytesIO()
    # Quality 90 keeps painterly textures clean; the default of 75 visibly blocks them
    stylyzed_image.save(buffer, format="JPEG", quality=90)

    return Response(content=buffer.getvalue(), media_type="image/jpeg")
