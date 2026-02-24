"""I/O utility functions."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def load_image(path: str | Path) -> NDArray[np.float32]:
    """
    Load image from file.

    Args:
        path: Path to image file

    Returns:
        Image array [H, W, 3] in RGB, float32, range [0, 1]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load with OpenCV (BGR)
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def save_image(
    image: NDArray[np.float32],
    path: str | Path,
    quality: int = 95,
) -> None:
    """
    Save image to file.

    Args:
        image: Image array [H, W, 3] in RGB, float32, range [0, 1]
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Denormalize to [0, 255]
    img = (image * 255).astype(np.uint8)

    # Convert RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
