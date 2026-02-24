"""U-Net model architectures for lens correction."""

from typing import List, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class GeometryAwareUNet(nn.Module):
    """
    U-Net architecture for lens distortion correction.

    Uses pretrained encoder (EfficientNet, ResNet, etc.) with U-Net decoder.
    Optionally includes attention mechanisms and distortion parameter estimation.

    Args:
        encoder_name: Name of encoder (from timm)
        encoder_weights: Pretrained weights ('imagenet' or None)
        decoder_channels: List of channel counts for decoder layers
        attention_type: Attention mechanism ('scse', 'cbam', or None)
        estimate_params: Whether to add distortion parameter regression head

    Example:
        >>> model = GeometryAwareUNet(
        ...     encoder_name='efficientnet-b4',
        ...     encoder_weights='imagenet',
        ...     attention_type='scse'
        ... )
        >>> x = torch.randn(2, 3, 512, 512)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 3, 512, 512])
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        attention_type: Optional[str] = None,
        estimate_params: bool = False,
    ) -> None:
        super().__init__()

        self.estimate_params = estimate_params

        # U-Net from segmentation_models_pytorch
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_attention_type=attention_type,
            in_channels=3,
            classes=3,  # RGB output
            activation=None,  # No activation (regression)
        )

        # Optional: Distortion parameter estimation head
        if estimate_params:
            # Get encoder output channels
            encoder_channels = self.unet.encoder.out_channels[-1]

            self.param_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 5),  # k1, k2, k3, p1, p2
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            If estimate_params=False:
                Corrected image [B, 3, H, W]
            If estimate_params=True:
                Tuple of (corrected_image, distortion_params)
                where distortion_params is [B, 5]
        """
        # Get features from encoder (for param estimation)
        if self.estimate_params:
            features = self.unet.encoder(x)
            params = self.param_head(features[-1])

        # U-Net forward
        corrected = self.unet(x)

        # Ensure output is in valid range [0, 1]
        corrected = torch.sigmoid(corrected)

        if self.estimate_params:
            return corrected, params

        return corrected


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for faster inference.

    Uses smaller encoder and fewer decoder channels.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: Optional[str] = "imagenet",
    ) -> None:
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=[128, 64, 32, 16],
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        corrected = self.unet(x)
        return torch.sigmoid(corrected)
