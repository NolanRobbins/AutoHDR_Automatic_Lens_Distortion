"""Geometric utility functions for distortion analysis."""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


def detect_lines(image: NDArray[np.uint8], **kwargs) -> List[NDArray[np.float32]]:
    """
    Detect lines in image using Hough transform.

    Args:
        image: Input image [H, W, 3] or [H, W]
        **kwargs: Arguments for cv2.HoughLinesP

    Returns:
        List of detected lines [x1, y1, x2, y2]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=kwargs.get("rho", 1),
        theta=kwargs.get("theta", np.pi / 180),
        threshold=kwargs.get("threshold", 100),
        minLineLength=kwargs.get("minLineLength", 50),
        maxLineGap=kwargs.get("maxLineGap", 10),
    )

    if lines is None:
        return []

    return [line[0] for line in lines]


def compute_line_curvature(line: NDArray[np.float32]) -> float:
    """
    Compute curvature of a line.

    For a straight line, curvature should be ~0.
    For distorted lines, curvature > 0.

    Args:
        line: Line coordinates [x1, y1, x2, y2]

    Returns:
        Curvature measure
    """
    # TODO: Implement proper curvature calculation
    # For now, return placeholder
    return 0.0
