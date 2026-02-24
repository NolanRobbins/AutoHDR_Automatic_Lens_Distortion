"""Evaluation metrics for lens correction."""

import torch
import kornia
from typing import Dict


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        max_val: Maximum possible pixel value

    Returns:
        PSNR in decibels
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]

    Returns:
        SSIM value in [0, 1]
    """
    # kornia.losses.ssim_loss returns 1 - SSIM
    ssim_loss = kornia.losses.ssim_loss(pred, target, window_size=11)
    ssim = 1 - ssim_loss
    return ssim.item()


def compute_geometric_score(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute approximation of competition's geometric score.

    This is a local approximation - the actual score comes from
    bounty.autohdr.com which uses proprietary metrics.

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]

    Returns:
        Dictionary with metric components
    """
    metrics = {}

    # PSNR
    metrics["psnr"] = compute_psnr(pred, target)

    # SSIM
    metrics["ssim"] = compute_ssim(pred, target)

    # Edge alignment (Sobel-based)
    pred_edges = kornia.filters.sobel(pred)
    target_edges = kornia.filters.sobel(target)
    edge_error = torch.mean(torch.abs(pred_edges - target_edges))
    metrics["edge_error"] = edge_error.item()

    # Gradient similarity
    pred_grad = kornia.filters.spatial_gradient(pred)
    target_grad = kornia.filters.spatial_gradient(target)
    grad_error = torch.mean(torch.abs(pred_grad - target_grad))
    metrics["gradient_error"] = grad_error.item()

    # Approximate overall score (weighted combination)
    # This is just an approximation - actual scoring happens externally
    score = (
        0.3 * metrics["ssim"]  # Structural similarity
        + 0.3 * (1 - metrics["edge_error"])  # Edge alignment
        + 0.2 * (metrics["psnr"] / 40.0)  # Pixel accuracy (normalized)
        + 0.2 * (1 - metrics["gradient_error"])  # Gradient orientation
    )
    metrics["approximate_score"] = max(0.0, min(100.0, score * 100))

    return metrics
