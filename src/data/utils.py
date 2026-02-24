"""Data utility functions."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


def compute_dataset_statistics(data_dir: str | Path) -> Dict[str, any]:
    """
    Compute statistics for the dataset.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)

    # Count samples
    if (data_dir / "train").exists():
        n_train = len(list((data_dir / "train").glob("*/")))
    else:
        n_train = 0

    if (data_dir / "test").exists():
        n_test = len(list((data_dir / "test").glob("*.jpg")))
    else:
        n_test = 0

    return {
        "n_train_pairs": n_train,
        "n_test_images": n_test,
        "total_samples": n_train + n_test,
    }


def stratified_split_by_distortion(
    samples: List[Dict],
    distortion_scores: NDArray[np.float32],
    n_bins: int = 5,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/val split based on distortion severity.

    Args:
        samples: List of sample dictionaries
        distortion_scores: Array of distortion severity scores [n_samples]
        n_bins: Number of severity bins for stratification
        val_split: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices)
    """
    rng = np.random.RandomState(seed)

    # Bin distortion scores
    bins = np.linspace(distortion_scores.min(), distortion_scores.max(), n_bins + 1)
    bin_indices = np.digitize(distortion_scores, bins)

    train_idx = []
    val_idx = []

    # Stratified split within each bin
    for bin_id in range(1, n_bins + 1):
        bin_mask = bin_indices == bin_id
        bin_samples = np.where(bin_mask)[0]

        # Shuffle within bin
        rng.shuffle(bin_samples)

        # Split
        n_val_in_bin = int(len(bin_samples) * val_split)
        val_idx.extend(bin_samples[:n_val_in_bin])
        train_idx.extend(bin_samples[n_val_in_bin:])

    return train_idx, val_idx
