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

        # Check which architecture to use based on config
        self.model_type = model_config.get("type", "unet")
        _loss_internal_keys = {"type", "w_param_reg"}
        loss_kwargs = {k: v for k, v in loss_config.items() if k not in _loss_internal_keys}

        if self.model_type == "radial":
            from src.models.swin_tps import RadialDistortionCorrector
            self.model = RadialDistortionCorrector(
                model_name=model_config.get("encoder", "swin_tiny_patch4_window7_224"),
                pretrained=model_config.get("pretrained", True),
                image_size=tuple(model_config.get("image_size", (224, 224))),
                num_k=model_config.get("num_k", 2),
                predict_center=model_config.get("predict_center", False),
            )
            from src.models.losses import TPSPerceptualLoss
            self.criterion = TPSPerceptualLoss(**loss_kwargs)
            self.w_param_reg = loss_config.get("w_param_reg", 0.0)

        elif self.model_type == "swin_tps":
            from src.models.swin_tps import SwinTPSLensCorrection
            self.model = SwinTPSLensCorrection(
                model_name=model_config.get("encoder", "swin_tiny_patch4_window7_224"),
                pretrained=model_config.get("pretrained", True),
                grid_size=model_config.get("grid_size", 10),
                image_size=tuple(model_config.get("image_size", (512, 512))),
            )
            from src.models.losses import TPSPerceptualLoss
            self.criterion = TPSPerceptualLoss(**loss_kwargs)
            self.w_param_reg = 0.0

        else:
            from src.models.unet import GeometryAwareUNet
            from src.models.losses import GeometricLoss
            self.model = GeometryAwareUNet(
                encoder_name=model_config.get("encoder", "efficientnet-b3"),
                encoder_weights=model_config.get("encoder_weights", "imagenet"),
                decoder_channels=model_config.get("decoder_channels", [256, 128, 64, 32, 16]),
                attention_type=model_config.get("attention", None),
                estimate_params=model_config.get("estimate_params", False),
            )
            self.criterion = GeometricLoss(**loss_kwargs)
            self.w_param_reg = 0.0

        # Store optimizer config
        self.optim_config = optim_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def _unpack_prediction(
        self, pred: Any, corrected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Unpack model output and compute loss for any model type."""
        if self.model_type in ("swin_tps", "radial"):
            pred_img, dense_grid, params = pred
            # TPS passes control_points for smoothness; radial passes None
            cp = params if self.model_type == "swin_tps" else None
            loss, loss_dict = self.criterion(pred_img, corrected, cp)
            # Optional L2 regularization on radial distortion params
            if self.w_param_reg > 0 and self.model_type == "radial":
                reg = torch.mean(params ** 2)
                loss = loss + self.w_param_reg * reg
                loss_dict["param_reg"] = reg.item()
        else:
            if isinstance(pred, tuple):
                pred_img, _ = pred
            else:
                pred_img = pred
            loss, loss_dict = self.criterion(pred_img, corrected)

        return pred_img, loss, loss_dict

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        pred = self(batch["distorted"])
        _, loss, loss_dict = self._unpack_prediction(pred, batch["corrected"])

        self.log("train_loss", loss, prog_bar=True)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        with torch.no_grad():
            pred = self(batch["distorted"])
            pred_img, loss, loss_dict = self._unpack_prediction(
                pred, batch["corrected"]
            )

        psnr = compute_psnr(pred_img, batch["corrected"])
        ssim = compute_ssim(pred_img, batch["corrected"])

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
