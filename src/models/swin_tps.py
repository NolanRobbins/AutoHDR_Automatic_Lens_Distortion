"""Swin-T based architectures for lens distortion correction.

Provides two correction strategies:
  1. TPS (Thin-Plate Spline): General-purpose grid deformation via control points.
  2. Radial: Physics-based Brown-Conrady model predicting k1/k2 distortion coefficients.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TPSGridGenerator(nn.Module):
    """
    Thin-Plate Spline (TPS) Grid Generator.
    Computes a dense displacement field from a sparse set of control points.
    """
    def __init__(self, target_size: Tuple[int, int], grid_size: int = 10):
        super().__init__()
        self.target_size = target_size
        self.grid_size = grid_size
        
        # Create a regular grid of source control points [-1, 1]
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # grid_sample expects (x, y) coordinates
        # [num_points, 2]
        self.source_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        self.num_points = grid_size * grid_size
        
        # Compute inverse of L matrix for TPS interpolation
        # L = [K, P; P^T, 0]
        K = self._compute_k_matrix(self.source_points, self.source_points)
        P = torch.cat([torch.ones(self.num_points, 1), self.source_points], dim=1)
        
        L = torch.zeros(self.num_points + 3, self.num_points + 3)
        L[:self.num_points, :self.num_points] = K
        L[:self.num_points, self.num_points:] = P
        L[self.num_points:, :self.num_points] = P.t()
        
        # We add a small ridge penalty for numerical stability
        L[:self.num_points, :self.num_points] += 1e-4 * torch.eye(self.num_points)
        
        # When padding=1 in reflection padding, we need a slightly larger grid
        # But to fix the "ugly stretching", we should set padding_mode="zeros"
        # or handle grid values outside [-1, 1].
        # For now, let's keep it simple.
        self.register_buffer('L_inv', torch.linalg.inv(L))
        
        # Create the dense target grid [-1, 1]
        H, W = target_size
        ty = torch.linspace(-1, 1, H)
        tx = torch.linspace(-1, 1, W)
        tyy, txx = torch.meshgrid(ty, tx, indexing='ij')
        
        # [H*W, 2] -> (x, y) coordinates for grid_sample
        self.target_points = torch.stack([txx.flatten(), tyy.flatten()], dim=1)
        
        # Precompute K for target points vs source points
        self.register_buffer('K_target', self._compute_k_matrix(self.target_points, self.source_points))
        self.register_buffer('P_target', torch.cat([torch.ones(H * W, 1), self.target_points], dim=1))

    def _compute_k_matrix(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Computes the TPS kernel matrix K(p1, p2) = U(r) = r^2 log(r^2)"""
        # p1: [N1, 2], p2: [N2, 2]
        # dists: [N1, N2]
        dists = torch.cdist(p1, p2)
        r_sq = dists.pow(2)
        # Handle zero distances to avoid log(0)
        r_sq = r_sq + 1e-8
        K = r_sq * torch.log(r_sq)
        # Set self-distances exactly to 0
        K[dists < 1e-6] = 0
        return K

    def forward(self, target_control_points: torch.Tensor) -> torch.Tensor:
        """
        Compute dense grid from target control points.
        
        Args:
            target_control_points: [B, num_points, 2]
            
        Returns:
            dense_grid: [B, H, W, 2] ready for F.grid_sample
        """
        B = target_control_points.shape[0]
        
        # Y = [V; 0]
        # V is target_control_points [B, num_points, 2]
        Y = torch.cat([
            target_control_points, 
            torch.zeros(B, 3, 2, device=target_control_points.device)
        ], dim=1) # [B, num_points + 3, 2]
        
        # W = L_inv * Y
        # L_inv is [num_points+3, num_points+3]
        W = torch.matmul(self.L_inv.unsqueeze(0), Y) # [B, num_points+3, 2]
        
        # Split W into W_1 (weights) and W_2 (affine params)
        W_1 = W[:, :self.num_points, :] # [B, num_points, 2]
        W_2 = W[:, self.num_points:, :] # [B, 3, 2]
        
        # f(x,y) = a1 + a2*x + a3*y + sum(w_i * U(|(x,y) - (x_i, y_i)|))
        # K_target: [H*W, num_points]
        # P_target: [H*W, 3]
        non_affine = torch.matmul(self.K_target.unsqueeze(0), W_1) # [B, H*W, 2]
        affine = torch.matmul(self.P_target.unsqueeze(0), W_2) # [B, H*W, 2]
        
        dense_points = non_affine + affine # [B, H*W, 2]
        
        # Reshape to [B, H, W, 2]
        H, W_dim = self.target_size
        return dense_points.view(B, H, W_dim, 2)


class SwinTPSLensCorrection(nn.Module):
    """
    Swin-T based architecture that predicts TPS control points 
    to unwarp the input image without generating RGB pixels.
    """
    def __init__(
        self, 
        model_name: str = 'swin_tiny_patch4_window7_224', 
        pretrained: bool = True,
        grid_size: int = 10,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()
        
        # 1. Swin-T Encoder
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        
        # Get the feature dimension (usually 768 for Swin-T)
        feature_dim = self.encoder.num_features
        
        # Number of control points * 2 (X and Y coordinates)
        self.num_points = grid_size * grid_size
        out_features = self.num_points * 2
        
        # 2. Control Point Predictor (MLP head)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_features)
        )
        
        # Initialize the final layer so that the network starts by predicting the identity grid
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        
        # 3. TPS Grid Generator
        self.tps = TPSGridGenerator(target_size=image_size, grid_size=grid_size)
        
        # Register the source points so we can add the delta to them
        self.register_buffer('identity_control_points', self.tps.source_points.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Tuple of:
            - corrected_image: The high-res unwarped image [B, 3, H, W]
            - dense_grid: The predicted flow field [B, H, W, 2]
            - control_points: The predicted TPS control points [B, num_points, 2]
        """
        B = x.shape[0]
        
        # 1. Extract global features
        features = self.encoder(x) # [B, feature_dim]
        
        # 2. Predict control point *deltas*
        delta = self.predictor(features) # [B, num_points * 2]
        delta = delta.view(B, self.num_points, 2)
        
        # Add delta to identity grid
        control_points = self.identity_control_points.unsqueeze(0) + delta
        
        # 3. Generate dense sampling grid
        dense_grid = self.tps(control_points) # [B, H, W, 2]
        
        # 4. Resample original image
        # mode='bilinear' is safer than bicubic for large distortions to prevent artifacting.
        # padding_mode='zeros' prevents the ugly edge repeating/reflection
        corrected_image = F.grid_sample(x, dense_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return corrected_image, dense_grid, control_points


class RadialDistortionCorrector(nn.Module):
    """
    Swin-T encoder that predicts radial distortion coefficients (k1, k2)
    and analytically generates a correction grid via the Brown-Conrady model.

    Instead of 200 free TPS parameters, this predicts 2-4 physically meaningful
    coefficients, giving the optimizer a much simpler loss landscape.

    Args:
        model_name: timm model identifier for the encoder
        pretrained: Whether to load pretrained encoder weights
        image_size: (H, W) of input images
        num_k: Number of radial distortion coefficients (1=k1 only, 2=k1+k2)
        predict_center: If True, also predict distortion center (cx, cy)
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        num_k: int = 2,
        predict_center: bool = False,
    ) -> None:
        super().__init__()
        self.num_k = num_k
        self.predict_center = predict_center
        num_params = num_k + (2 if predict_center else 0)

        # Swin-T encoder (same as TPS variant)
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.encoder.num_features

        # Lightweight head: 768-d → 2-4 scalars
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_params),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

        # Precompute normalized coordinate grid [-1, 1]
        H, W = image_size
        gy = torch.linspace(-1, 1, H)
        gx = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        self.register_buffer("base_grid_x", grid_x.clone())  # [H, W]
        self.register_buffer("base_grid_y", grid_y.clone())  # [H, W]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Distorted input image [B, 3, H, W]

        Returns:
            Tuple of:
            - corrected_image: Unwarped image [B, 3, H, W]
            - dense_grid: Sampling grid [B, H, W, 2]
            - params: Predicted distortion parameters [B, num_params]
        """
        B = x.shape[0]

        features = self.encoder(x)
        params = self.predictor(features)  # [B, num_params]

        # Split parameters
        k1 = params[:, 0].view(B, 1, 1)
        k2 = params[:, 1].view(B, 1, 1) if self.num_k >= 2 else 0.0

        if self.predict_center:
            cx = params[:, self.num_k].view(B, 1, 1)
            cy = params[:, self.num_k + 1].view(B, 1, 1)
        else:
            cx, cy = 0.0, 0.0

        # Shift grid relative to predicted (or assumed) distortion center
        dx = self.base_grid_x.unsqueeze(0) - cx  # [B, H, W]
        dy = self.base_grid_y.unsqueeze(0) - cy

        # Brown-Conrady radial distortion model:
        #   r² = dx² + dy²
        #   scale = 1 + k1·r² + k2·r⁴
        #   (x_d, y_d) = (cx + dx·scale, cy + dy·scale)
        r_sq = dx ** 2 + dy ** 2
        scale = 1.0 + k1 * r_sq + k2 * r_sq ** 2  # [B, H, W]

        grid_x = cx + dx * scale
        grid_y = cy + dy * scale
        dense_grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        corrected_image = F.grid_sample(
            x, dense_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return corrected_image, dense_grid, params
