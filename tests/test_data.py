"""Tests for data module."""

import pytest
import torch
from pathlib import Path


def test_dataset_import() -> None:
    """Test that dataset can be imported."""
    from src.data import LensDistortionDataset

    assert LensDistortionDataset is not None


def test_transforms_import() -> None:
    """Test that transforms can be imported."""
    from src.data import get_train_transforms, get_val_transforms, get_test_transforms

    assert get_train_transforms is not None
    assert get_val_transforms is not None
    assert get_test_transforms is not None


# TODO: Add more comprehensive tests once data is available
