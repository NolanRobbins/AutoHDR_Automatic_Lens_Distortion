"""Loss functions for lens distortion correction."""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

try:
    import lpips
except ImportError:
    lpips = None


class TPSPerceptualLoss(nn.Module):
    """
    Perceptual and Edge-Aware Loss for TPS-based Lens Correction.

    Combines L1 pixel loss (stable gradients), LPIPS perceptual loss (human-aligned),
    Sobel edge loss (straight-line preservation), and TPS smoothness regularization.

    Args:
        w_l1: Weight for L1 pixel loss (default: 0.0, set >0 to enable)
        w_lpips: Weight for LPIPS perceptual loss (default: 1.0)
        w_edge: Weight for Sobel-based edge alignment/straightness loss (default: 3.0)
        w_tps_smooth: Weight for TPS control point smoothness regularization (default: 0.5)
    """

    def __init__(
        self,
        w_l1: float = 0.0,
        w_lpips: float = 1.0,
        w_edge: float = 3.0,
        w_tps_smooth: float = 0.5,
    ) -> None:
        super().__init__()

        self.w_l1 = w_l1
        self.w_lpips = w_lpips
        self.w_edge = w_edge
        self.w_tps_smooth = w_tps_smooth

        if lpips is None and w_lpips > 0:
            raise ImportError(
                "LPIPS library is required for TPSPerceptualLoss. "
                "Install it with `pip install lpips`."
            )

        if w_lpips > 0:
            self.lpips_vgg = lpips.LPIPS(net="vgg")
            for param in self.lpips_vgg.parameters():
                param.requires_grad = False

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        control_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual components.

        Args:
            pred: Predicted corrected image [B, C, H, W]
            target: Ground truth corrected image [B, C, H, W]
            control_points: Predicted TPS control points [B, num_points, 2]

        Returns:
            Tuple of (total_loss, component_dict)
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        # 1. L1 Pixel Loss (provides stable, dense gradient signal)
        if self.w_l1 > 0:
            l_l1 = F.l1_loss(pred, target)
            losses["l1"] = l_l1.item()
            total += self.w_l1 * l_l1

        # 2. Perceptual Loss (LPIPS)
        if self.w_lpips > 0:
            pred_scaled = pred * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            l_lpips = self.lpips_vgg(pred_scaled, target_scaled).mean()
            losses["lpips"] = l_lpips.item()
            total += self.w_lpips * l_lpips

        # 3. Edge / Plumb-Line Loss (Sobel-based)
        if self.w_edge > 0:
            l_edge = self._edge_loss(pred, target)
            losses["edge"] = l_edge.item()
            total += self.w_edge * l_edge

        # 4. TPS Smoothness Regularization
        if self.w_tps_smooth > 0 and control_points is not None:
            l_tps_smooth = self._tps_smoothness_loss(control_points)
            losses["tps_smooth"] = l_tps_smooth.item()
            total += self.w_tps_smooth * l_tps_smooth
            
        losses["total"] = total.item()

        return total, losses

    def _edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge alignment loss using Sobel filter.
        Heavily penalizes differences in structural edges (straight lines).
        """
        # Convert to grayscale for edge detection
        pred_gray = kornia.color.rgb_to_grayscale(pred)
        target_gray = kornia.color.rgb_to_grayscale(target)
        
        pred_edges = kornia.filters.sobel(pred_gray)
        target_edges = kornia.filters.sobel(target_gray)

        return F.l1_loss(pred_edges, target_edges)

    def _tps_smoothness_loss(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Regularizes the predicted control points to form a smooth, non-crossing grid.
        Minimizes the second-order derivative (bending energy) of the control points.
        
        Args:
            control_points: [B, grid_size * grid_size, 2]
        """
        B, num_points, _ = control_points.shape
        grid_size = int(num_points ** 0.5)
        
        # Reshape to a 2D grid of control points [B, grid_size, grid_size, 2]
        grid = control_points.view(B, grid_size, grid_size, 2)
        
        # Calculate second-order finite differences
        # Horizontal smoothness (along W)
        dxx = grid[:, :, 2:, :] - 2 * grid[:, :, 1:-1, :] + grid[:, :, :-2, :]
        # Vertical smoothness (along H)
        dyy = grid[:, 2:, :, :] - 2 * grid[:, 1:-1, :, :] + grid[:, :-2, :, :]
        # Diagonal smoothness
        dxy = grid[:, 1:, 1:, :] - grid[:, 1:, :-1, :] - grid[:, :-1, 1:, :] + grid[:, :-1, :-1, :]
        
        loss = torch.mean(dxx**2) + torch.mean(dyy**2) + 2 * torch.mean(dxy**2)
        return loss


class GeometricLoss(nn.Module):
    """Legacy Geometric Loss for U-Net Baseline"""
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
        losses = {}

        if self.w_pixel > 0:
            l_pixel = F.l1_loss(pred, target)
            losses["pixel"] = l_pixel.item()
        else:
            l_pixel = 0.0

        if self.w_edge > 0:
            l_edge = self._edge_loss(pred, target)
            losses["edge"] = l_edge.item()
        else:
            l_edge = 0.0

        if self.w_ssim > 0:
            l_ssim = self._ssim_loss(pred, target)
            losses["ssim"] = l_ssim.item()
        else:
            l_ssim = 0.0

        if self.w_gradient > 0:
            l_gradient = self._gradient_loss(pred, target)
            losses["gradient"] = l_gradient.item()
        else:
            l_gradient = 0.0

        total = (
            self.w_pixel * l_pixel
            + self.w_edge * l_edge
            + self.w_ssim * l_ssim
            + self.w_gradient * l_gradient
        )

        return total, losses

    def _edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = kornia.filters.sobel(pred)
        target_edges = kornia.filters.sobel(target)
        return F.l1_loss(pred_edges, target_edges)

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return kornia.losses.ssim_loss(pred, target, window_size=11)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = kornia.filters.spatial_gradient(pred, order=1)
        target_grad = kornia.filters.spatial_gradient(target, order=1)
        return F.l1_loss(pred_grad, target_grad)


class CombinedLoss(nn.Module):
    """Simple combined loss (L1 + SSIM) for baseline."""
    def __init__(self, w_l1: float = 1.0, w_ssim: float = 1.0) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = F.l1_loss(pred, target)
        ssim = kornia.losses.ssim_loss(pred, target, window_size=11)
        total = self.w_l1 * l1 + self.w_ssim * ssim
        return total, {"l1": l1.item(), "ssim": ssim.item()}
