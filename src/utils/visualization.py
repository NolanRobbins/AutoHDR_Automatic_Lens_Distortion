"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_comparison(
    distorted: np.ndarray,
    corrected: np.ndarray,
    predicted: Optional[np.ndarray] = None,
    title: str = "Lens Correction Comparison",
) -> plt.Figure:
    """
    Plot side-by-side comparison of images.

    Args:
        distorted: Distorted input image
        corrected: Ground truth corrected image
        predicted: Predicted corrected image (optional)
        title: Figure title

    Returns:
        Matplotlib figure
    """
    n_cols = 3 if predicted is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(distorted)
    axes[0].set_title("Distorted")
    axes[0].axis("off")

    axes[1].imshow(corrected)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if predicted is not None:
        axes[2].imshow(predicted)
        axes[2].set_title("Predicted")
        axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_distortion_field(
    distorted: np.ndarray,
    corrected: np.ndarray,
    step: int = 32,
) -> plt.Figure:
    """
    Plot distortion vector field showing geometric transformation.

    Args:
        distorted: Distorted image
        corrected: Corrected image
        step: Grid step size for vector field

    Returns:
        Matplotlib figure
    """
    # TODO: Implement optical flow visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(distorted)
    ax.set_title("Distortion Field (placeholder)")
    return fig
