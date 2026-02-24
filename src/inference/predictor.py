"""Inference pipeline for test set prediction."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class LensPredictor:
    """
    Predictor class for lens distortion correction inference.

    Args:
        model: Trained model
        device: Device to run inference on
        tta: Whether to use test-time augmentation

    Example:
        >>> from src.models import GeometryAwareUNet
        >>> model = GeometryAwareUNet()
        >>> predictor = LensPredictor(model, device='cuda', tta=True)
        >>> corrected = predictor.predict_image('test/0001.jpg')
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        tta: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.tta = tta

    @torch.no_grad()
    def predict_image(self, image_path: str | Path) -> torch.Tensor:
        """
        Predict corrected image for a single input.

        Args:
            image_path: Path to distorted image

        Returns:
            Corrected image tensor
        """
        # TODO: Implement full inference pipeline
        # 1. Load image
        # 2. Preprocess
        # 3. Model forward (with TTA if enabled)
        # 4. Post-process
        # 5. Return corrected image

        pass

    def predict_batch(self, image_paths: list[str | Path]) -> list[torch.Tensor]:
        """Predict for multiple images."""
        return [self.predict_image(path) for path in image_paths]
