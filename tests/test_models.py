"""Tests for model architectures."""

import pytest
import torch


def test_unet_import() -> None:
    """Test that U-Net can be imported."""
    from src.models import GeometryAwareUNet

    assert GeometryAwareUNet is not None


def test_unet_forward() -> None:
    """Test U-Net forward pass."""
    from src.models import GeometryAwareUNet

    model = GeometryAwareUNet(
        encoder_name="efficientnet-b0",  # Small for testing
        encoder_weights=None,  # Don't download weights for test
    )

    # Create dummy input
    x = torch.randn(2, 3, 256, 256)

    # Forward pass
    out = model(x)

    # Check output shape
    assert out.shape == (2, 3, 256, 256)
    assert torch.all((out >= 0) & (out <= 1))  # Should be in [0, 1]


def test_loss_import() -> None:
    """Test that losses can be imported."""
    from src.models import GeometricLoss

    assert GeometricLoss is not None


def test_geometric_loss() -> None:
    """Test geometric loss computation."""
    from src.models import GeometricLoss

    loss_fn = GeometricLoss()

    pred = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)

    loss, components = loss_fn(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert isinstance(components, dict)
    assert "pixel" in components
