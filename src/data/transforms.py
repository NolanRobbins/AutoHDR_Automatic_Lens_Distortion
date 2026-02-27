"""Data augmentation and preprocessing transforms."""

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    size: Tuple[int, int] = (512, 512),
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    hue: float = 0.0,
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        size: Target image size (height, width)
        brightness: Max brightness adjustment
        contrast: Max contrast adjustment
        saturation: Max saturation adjustment
        hue: Max hue adjustment

    Returns:
        Albumations compose object with training transforms

    Note:
        Avoids geometric transforms (rotation, perspective) that would
        interfere with learning lens distortion patterns.
    """
    return A.Compose(
        [
            # Spatial transforms (safe for distortion learning)
            A.RandomCrop(height=size[0], width=size[1], p=1.0),
            A.HorizontalFlip(p=0.5),

            # Photometric transforms (safe - metric focuses on geometry)
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5,
            ),

            # Noise and blur (mild, helps generalization)
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
        ],
        # Enable consistent transform for both distorted and corrected
        additional_targets={"mask": "image"},
    )


def get_val_transforms(size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get validation preprocessing pipeline.

    Args:
        size: Target image size (height, width)

    Returns:
        Albumentations compose object with validation transforms

    Note:
        No augmentation, only resize and normalize.
    """
    return A.Compose(
        [
            A.Resize(height=size[0], width=size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        additional_targets={"mask": "image"},
    )


def get_test_transforms(size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get test preprocessing pipeline.

    Args:
        size: Target image size (height, width)

    Returns:
        Albumentations compose object with test transforms
    """
    return A.Compose(
        [
            A.Resize(height=size[0], width=size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_advanced_train_transforms(
    size: Tuple[int, int] = (512, 512),
) -> A.Compose:
    """
    Get advanced training augmentation with more aggressive transforms.

    Args:
        size: Target image size (height, width)

    Returns:
        Albumentations compose object with advanced augmentations
    """
    return A.Compose(
        [
            # Spatial
            A.RandomCrop(height=size[0], width=size[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),

            # Photometric (more aggressive)
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=0.6,
            ),

            # Advanced photometric
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ],
                p=0.3,
            ),

            # Noise and blur
            A.GaussNoise(p=0.4),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        additional_targets={"mask": "image"},
    )
