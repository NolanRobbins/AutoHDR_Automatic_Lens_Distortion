"""Tests for loss functions."""

import pytest
import torch


def test_combined_loss() -> None:
    """Test combined loss."""
    from src.models.losses import CombinedLoss

    loss_fn = CombinedLoss(w_l1=1.0, w_ssim=1.0)

    pred = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 3, 128, 128)

    loss, components = loss_fn(pred, target)

    assert loss.item() >= 0
    assert "l1" in components
    assert "ssim" in components


# TODO: Add more comprehensive loss tests
