"""Classical per-image k1 radial distortion optimizer.

Uses Canny edge detection + Hough line detection to find straight lines,
then optimizes k1 to maximize line straightness via scipy.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


def undistort_points(
    points: NDArray[np.float64],
    k1: float,
    k2: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> NDArray[np.float64]:
    """Apply radial undistortion to a set of 2D points in normalized coords.

    Args:
        points: [N, 2] array of (x, y) coordinates in [-1, 1]
        k1: Primary radial distortion coefficient
        k2: Secondary radial distortion coefficient
        cx: Distortion center x offset
        cy: Distortion center y offset

    Returns:
        Undistorted points [N, 2]
    """
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    r_sq = dx ** 2 + dy ** 2
    scale = 1.0 + k1 * r_sq + k2 * r_sq ** 2
    out = np.empty_like(points)
    out[:, 0] = cx + dx * scale
    out[:, 1] = cy + dy * scale
    return out


def _detect_lines(
    image: NDArray[np.uint8],
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 60,
    min_line_length: int = 40,
    max_line_gap: int = 10,
) -> NDArray[np.int32]:
    """Detect line segments using Canny + probabilistic Hough transform.

    Args:
        image: Grayscale uint8 image [H, W]

    Returns:
        Line segments [N, 4] where each row is (x1, y1, x2, y2)
    """
    edges = cv2.Canny(image, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def _line_straightness_score(
    lines: NDArray[np.int32],
    k1: float,
    height: int,
    width: int,
    k2: float = 0.0,
) -> float:
    """Score how straight detected lines remain after applying radial correction.

    Lower score = straighter lines = better correction.
    For each line segment, we sample intermediate points along it, apply the
    radial model, then measure deviation from the straight line connecting
    the corrected endpoints.

    Args:
        lines: [N, 4] line segments (x1, y1, x2, y2) in pixel coords
        k1: Radial distortion coefficient to evaluate
        height: Image height
        width: Image width
        k2: Secondary coefficient

    Returns:
        Mean squared deviation from straightness (lower = better)
    """
    if len(lines) == 0:
        return 0.0

    num_samples = 10
    total_deviation = 0.0
    count = 0

    half_w = width / 2.0
    half_h = height / 2.0

    for x1, y1, x2, y2 in lines:
        t = np.linspace(0, 1, num_samples + 2)
        pts_x = x1 + t * (x2 - x1)
        pts_y = y1 + t * (y2 - y1)

        norm_pts = np.stack(
            [(pts_x - half_w) / half_w, (pts_y - half_h) / half_h], axis=1
        )

        corrected = undistort_points(norm_pts, k1, k2)

        p0 = corrected[0]
        p1 = corrected[-1]
        line_vec = p1 - p0
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq < 1e-12:
            continue

        for i in range(1, len(corrected) - 1):
            v = corrected[i] - p0
            proj = np.dot(v, line_vec) / line_len_sq
            closest = p0 + proj * line_vec
            deviation = np.sum((corrected[i] - closest) ** 2)
            total_deviation += deviation
            count += 1

    return total_deviation / max(count, 1)


def optimize_k1(
    image: NDArray[np.uint8],
    k1_range: Tuple[float, float] = (-0.5, 0.5),
    k2: float = 0.0,
    initial_k1: Optional[float] = None,
) -> Tuple[float, float, int]:
    """Find the optimal k1 for a single image via line straightness optimization.

    Args:
        image: RGB or BGR uint8 image [H, W, 3] or grayscale [H, W]
        k1_range: Search bounds for k1
        k2: Fixed k2 value (not optimized)
        initial_k1: Optional starting point (e.g. from DL model)

    Returns:
        Tuple of (best_k1, best_score, num_lines_detected)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    lines = _detect_lines(gray)
    num_lines = len(lines)

    if num_lines < 3:
        return initial_k1 if initial_k1 is not None else 0.0, 0.0, num_lines

    def objective(k1: float) -> float:
        return _line_straightness_score(lines, k1, h, w, k2)

    if initial_k1 is not None:
        bracket_width = 0.15
        bounds = (
            max(k1_range[0], initial_k1 - bracket_width),
            min(k1_range[1], initial_k1 + bracket_width),
        )
    else:
        bounds = k1_range

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x), float(result.fun), num_lines


def apply_radial_correction(
    image: NDArray[np.uint8],
    k1: float,
    k2: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> NDArray[np.uint8]:
    """Apply radial distortion correction to an image using OpenCV remap.

    Args:
        image: Input image [H, W, 3] uint8
        k1: Primary radial coefficient
        k2: Secondary radial coefficient
        cx: Center x offset in normalized [-1, 1]
        cy: Center y offset in normalized [-1, 1]

    Returns:
        Corrected image [H, W, 3] uint8
    """
    h, w = image.shape[:2]
    half_w, half_h = w / 2.0, h / 2.0

    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)

    nx = (map_x - half_w) / half_w - cx
    ny = (map_y - half_h) / half_h - cy
    r_sq = nx ** 2 + ny ** 2
    scale = 1.0 + k1 * r_sq + k2 * r_sq ** 2

    map_x_dst = (cx + nx * scale) * half_w + half_w
    map_y_dst = (cy + ny * scale) * half_h + half_h

    corrected = cv2.remap(
        image,
        map_x_dst.astype(np.float32),
        map_y_dst.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected
