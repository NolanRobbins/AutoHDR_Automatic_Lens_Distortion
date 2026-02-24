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
            # Training data: paired images in subdirectories
            pair_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

            for pair_dir in pair_dirs:
                original_path = pair_dir / "original.jpg"
                generated_path = pair_dir / "generated.jpg"

                if original_path.exists() and generated_path.exists():
                    samples.append({
                        "distorted_path": str(original_path),
                        "corrected_path": str(generated_path),
                        "pair_id": pair_dir.name,
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

        # Load distorted image
        distorted = self._load_image(sample["distorted_path"])

        # Build output dictionary
        output = {
            "distorted": distorted,
            "image_id": sample.get("pair_id", sample.get("image_id")),
        }

        # Load corrected image if available (training/val)
        if "corrected_path" in sample:
            corrected = self._load_image(sample["corrected_path"])

            # Apply transforms to both images consistently
            if self.transform is not None:
                # Albumentations expects numpy arrays
                distorted_np = distorted.permute(1, 2, 0).numpy()
                corrected_np = corrected.permute(1, 2, 0).numpy()

                # Use additional_targets to apply same transform
                transformed = self.transform(
                    image=distorted_np,
                    mask=corrected_np  # Trick: use mask for second image
                )

                distorted = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
                corrected = torch.from_numpy(transformed["mask"]).permute(2, 0, 1)

            output["corrected"] = corrected
        else:
            # Test set: only transform distorted
            if self.transform is not None:
                distorted_np = distorted.permute(1, 2, 0).numpy()
                transformed = self.transform(image=distorted_np)
                distorted = torch.from_numpy(transformed["image"]).permute(2, 0, 1)

            output["distorted"] = distorted

        return output

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load an image from disk.

        Args:
            path: Path to image file

        Returns:
            Image tensor [C, H, W] in range [0, 1]
        """
        # Load with OpenCV (BGR)
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor
