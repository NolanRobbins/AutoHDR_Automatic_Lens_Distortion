"""Dataset module for lens distortion correction."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.typing import NDArray


class LensDistortionDataset(Dataset):
    """
    Dataset for lens distortion correction.

    Supports both paired training data (distorted + corrected) and
    unpaired test data (distorted only).

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to apply to images
        val_split: Fraction of data to use for validation (if split='val')
        seed: Random seed for reproducibility

    Example:
        >>> from src.data import LensDistortionDataset, get_train_transforms
        >>> dataset = LensDistortionDataset(
        ...     data_dir="data/train",
        ...     split="train",
        ...     transform=get_train_transforms(size=(512, 512))
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['distorted', 'corrected', 'image_id'])
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.val_split = val_split
        self.seed = seed

        # Load image pairs/paths
        self.samples = self._load_samples()

        # Split train/val if needed
        if split in ("train", "val"):
            self.samples = self._split_train_val(self.samples)

    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load all samples from data directory.

        Returns:
            List of sample dictionaries with paths and metadata
        """
        samples = []

        if self.split in ("train", "val"):
            # Training data: pairs of *_original.jpg and *_generated.jpg
            original_files = sorted(list(self.data_dir.glob("*_original.jpg")))
            
            for orig_path in original_files:
                # The generated file has the same prefix but ends in _generated.jpg
                gen_path = self.data_dir / orig_path.name.replace("_original.jpg", "_generated.jpg")
                
                if gen_path.exists():
                    samples.append({
                        "distorted_path": str(orig_path),
                        "corrected_path": str(gen_path),
                        "pair_id": orig_path.stem.replace("_original", ""),
                    })
        else:
            # Test data: single images
            test_images = sorted(self.data_dir.glob("*.jpg"))

            for img_path in test_images:
                samples.append({
                    "distorted_path": str(img_path),
                    "image_id": img_path.stem,
                })

        return samples

    def _split_train_val(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split samples into train/val based on val_split.

        Args:
            samples: List of all samples

        Returns:
            Filtered list for current split
        """
        rng = np.random.RandomState(self.seed)
        n_samples = len(samples)
        n_val = int(n_samples * self.val_split)

        # Shuffle indices
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        # Split
        if self.split == "val":
            selected_indices = indices[:n_val]
        else:  # train
            selected_indices = indices[n_val:]

        return [samples[i] for i in selected_indices]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - distorted: Distorted image tensor [C, H, W]
                - corrected: Corrected image tensor [C, H, W] (if available)
                - image_id: Image identifier
        """
        sample = self.samples[idx]

        # Load images as uint8 numpy arrays [H, W, C] in [0, 255]
        distorted_np = self._load_image(sample["distorted_path"])

        # Build output dictionary
        output: Dict[str, torch.Tensor | str] = {
            "image_id": sample.get("pair_id", sample.get("image_id")),
        }

        if "corrected_path" in sample:
            corrected_np = self._load_image(sample["corrected_path"])

            # Ensure both images have the same dimensions
            target_h = min(distorted_np.shape[0], corrected_np.shape[0])
            target_w = min(distorted_np.shape[1], corrected_np.shape[1])
            distorted_np = distorted_np[:target_h, :target_w, :]
            corrected_np = corrected_np[:target_h, :target_w, :]

            if self.transform is not None:
                transformed = self.transform(
                    image=distorted_np,
                    mask=corrected_np,
                )
                distorted_np = transformed["image"]
                corrected_np = transformed["mask"]

            # Convert to tensors [H, W, C] -> [C, H, W]
            output["distorted"] = torch.from_numpy(distorted_np).permute(2, 0, 1).float()
            output["corrected"] = torch.from_numpy(corrected_np).permute(2, 0, 1).float()
        else:
            if self.transform is not None:
                transformed = self.transform(image=distorted_np)
                distorted_np = transformed["image"]

            output["distorted"] = torch.from_numpy(distorted_np).permute(2, 0, 1).float()

        return output

    def _load_image(self, path: str) -> NDArray[np.uint8]:
        """
        Load an image from disk as a uint8 numpy array.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array [H, W, C] in range [0, 255], uint8
        """
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
