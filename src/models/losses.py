"""Loss functions for lens distortion correction."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class GeometricLoss(nn.Module):
    """
    Multi-component geometric loss for lens distortion correction.

    Combines multiple loss components that align with the evaluation metric:
    - Pixel accuracy (L1)
    - Edge alignment (Sobel-based)
    - Structural similarity (SSIM)
    - Gradient orientation
    - Line straightness (optional, expensive)

    Args:
        w_pixel: Weight for pixel L1 loss
        w_edge: Weight for edge alignment loss
        w_ssim: Weight for SSIM loss
        w_gradient: Weight for gradient orientation loss
        w_line: Weight for line straightness loss (set to 0 to disable)

    Example:
        >>> loss_fn = GeometricLoss(w_pixel=1.0, w_edge=2.0, w_ssim=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss, components = loss_fn(pred, target)
        >>> print(f"Total loss: {loss.item():.4f}")
        >>> print(f"Components: {components}")
    """

    def __init__(
        self,
        w_pixel: float = 1.0,
        w_edge: float = 2.0,
        w_ssim: float = 1.0,
        w_gradient: float = 1.5,
        w_line: float = 0.0,
    ) -> None:
        super().__init__()

        self.w_pixel = w_pixel
        self.w_edge = w_edge
        self.w_ssim = w_ssim
        self.w_gradient = w_gradient
        self.w_line = w_line

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual components.

        Args:
            pred: Predicted corrected image [B, C, H, W]
            target: Ground truth corrected image [B, C, H, W]

        Returns:
            Tuple of (total_loss, component_dict)
        """
        losses = {}

        # 1. Pixel accuracy (L1 loss)
        if self.w_pixel > 0:
            l_pixel = F.l1_loss(pred, target)
            losses["pixel"] = l_pixel.item()
        else:
            l_pixel = 0.0

        # 2. Edge alignment
        if self.w_edge > 0:
            l_edge = self._edge_loss(pred, target)
            losses["edge"] = l_edge.item()
        else:
            l_edge = 0.0

        # 3. Structural similarity
        if self.w_ssim > 0:
            l_ssim = self._ssim_loss(pred, target)
            losses["ssim"] = l_ssim.item()
        else:
            l_ssim = 0.0

        # 4. Gradient orientation
        if self.w_gradient > 0:
            l_gradient = self._gradient_loss(pred, target)
            losses["gradient"] = l_gradient.item()
        else:
            l_gradient = 0.0

        # 5. Line straightness (expensive, usually disabled)
        if self.w_line > 0:
            l_line = self._line_straightness_loss(pred, target)
            losses["line"] = l_line.item()
        else:
            l_line = 0.0

        # Total weighted loss
        total = (
            self.w_pixel * l_pixel
            + self.w_edge * l_edge
            + self.w_ssim * l_ssim
            + self.w_gradient * l_gradient
            + self.w_line * l_line
        )

        return total, losses

    def _edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge alignment loss using Sobel filter.

        Measures difference in edge maps between pred and target.
        """
        pred_edges = kornia.filters.sobel(pred)
        target_edges = kornia.filters.sobel(target)

        return F.l1_loss(pred_edges, target_edges)

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Structural similarity loss.

        Returns 1 - SSIM so that minimizing loss maximizes SSIM.
        """
        ssim_val = kornia.losses.ssim_loss(pred, target, window_size=11)
        return ssim_val

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Gradient orientation loss.

        Measures difference in spatial gradients (directional derivatives).
        """
        # Compute spatial gradients
        pred_grad = kornia.filters.spatial_gradient(pred, order=1)
        target_grad = kornia.filters.spatial_gradient(target, order=1)

        # pred_grad and target_grad have shape [B, C, 2, H, W]
        # where [:, :, 0, :, :] is dx and [:, :, 1, :, :] is dy

        return F.l1_loss(pred_grad, target_grad)

    def _line_straightness_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Line straightness loss (placeholder).

        This is expensive to compute properly (requires line detection).
        For now, return zero. Can be implemented with differentiable
        Hough transform or orientation consistency measures.
        """
        # TODO: Implement if needed
        # Ideas:
        # 1. Differentiable Hough transform
        # 2. Local orientation variance
        # 3. Curvature penalty on high-gradient regions

        return torch.tensor(0.0, device=pred.device)


class CombinedLoss(nn.Module):
    """
    Simple combined loss (L1 + SSIM) for baseline.

    Args:
        w_l1: Weight for L1 loss
        w_ssim: Weight for SSIM loss
    """

    def __init__(self, w_l1: float = 1.0, w_ssim: float = 1.0) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        l1 = F.l1_loss(pred, target)
        ssim = kornia.losses.ssim_loss(pred, target, window_size=11)

        total = self.w_l1 * l1 + self.w_ssim * ssim

        return total, {"l1": l1.item(), "ssim": ssim.item()}
