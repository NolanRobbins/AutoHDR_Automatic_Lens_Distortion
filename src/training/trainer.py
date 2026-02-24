"""PyTorch Lightning training module."""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.models.unet import GeometryAwareUNet
from src.models.losses import GeometricLoss
from src.models.metrics import compute_psnr, compute_ssim


class LensCorrector(pl.LightningModule):
    """
    PyTorch Lightning module for lens distortion correction.

    Args:
        model_config: Model configuration dictionary
        loss_config: Loss function configuration dictionary
        optim_config: Optimizer configuration dictionary

    Example:
        >>> model_config = {
        ...     'encoder': 'efficientnet-b3',
        ...     'encoder_weights': 'imagenet',
        ... }
        >>> loss_config = {'w_pixel': 1.0, 'w_edge': 2.0, 'w_ssim': 1.0}
        >>> optim_config = {'lr': 1e-3, 'weight_decay': 1e-4}
        >>> module = LensCorrector(model_config, loss_config, optim_config)
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optim_config: Dict[str, Any],
    ) -> None:
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Build model
        self.model = GeometryAwareUNet(
            encoder_name=model_config.get("encoder", "efficientnet-b3"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            decoder_channels=model_config.get("decoder_channels", [256, 128, 64, 32, 16]),
            attention_type=model_config.get("attention", None),
            estimate_params=model_config.get("estimate_params", False),
        )

        # Loss function
        self.criterion = GeometricLoss(**loss_config)

        # Store optimizer config
        self.optim_config = optim_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        distorted = batch["distorted"]
        corrected = batch["corrected"]

        # Forward
        pred = self(distorted)

        # Handle multi-output (if estimate_params=True)
        if isinstance(pred, tuple):
            pred_img, pred_params = pred
        else:
            pred_img = pred

        # Loss
        loss, loss_dict = self.criterion(pred_img, corrected)

        # Log losses
        self.log("train_loss", loss, prog_bar=True)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        distorted = batch["distorted"]
        corrected = batch["corrected"]

        # Forward
        with torch.no_grad():
            pred = self(distorted)

            if isinstance(pred, tuple):
                pred_img, _ = pred
            else:
                pred_img = pred

        # Loss
        loss, loss_dict = self.criterion(pred_img, corrected)

        # Metrics
        psnr = compute_psnr(pred_img, corrected)
        ssim = compute_ssim(pred_img, corrected)

        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        self.log("val_ssim", ssim, prog_bar=True)

        for key, value in loss_dict.items():
            self.log(f"val_{key}", value)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_config.get("lr", 1e-3),
            weight_decay=self.optim_config.get("weight_decay", 1e-4),
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.optim_config.get("T_0", 10),
            T_mult=self.optim_config.get("T_mult", 2),
            eta_min=self.optim_config.get("eta_min", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
