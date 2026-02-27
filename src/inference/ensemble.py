"""Cascade ensemble: DL radial prediction -> classical line-based refinement.

Pipeline:
  1. Swin-T predicts initial k1/k2 from image features (works on all images)
  2. Classical optimizer refines k1 using detected line straightness
     (only if enough lines found, otherwise keeps DL prediction)
  3. Applies the final k1/k2 correction at full resolution via OpenCV remap
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image

from src.data.transforms import get_test_transforms
from src.models.classical import apply_radial_correction, optimize_k1
from src.training.trainer import LensCorrector


class EnsembleLensCorrector:
    """Cascade ensemble combining DL prediction with classical refinement.

    Args:
        checkpoint_path: Path to trained radial model checkpoint
        config_path: Path to the config YAML used for training
        min_lines_for_refinement: Minimum Hough lines needed to trust
            the classical optimizer. Below this, keep DL prediction.
        refine_k2: If True, also refine k2 (not yet implemented, kept at DL value)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        min_lines_for_refinement: int = 5,
        refine_k2: bool = False,
    ) -> None:
        self.min_lines = min_lines_for_refinement
        self.refine_k2 = refine_k2

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.img_size = tuple(self.config["model"].get("image_size", (224, 224)))
        self.transform = get_test_transforms(size=self.img_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LensCorrector.load_from_checkpoint(
            checkpoint_path,
            model_config=self.config["model"],
            loss_config=self.config["loss"],
            optim_config=self.config["training"],
            map_location="cpu",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_params(self, image_np: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Run DL model to get initial k1/k2 prediction.

        Args:
            image_np: RGB uint8 image [H, W, 3]

        Returns:
            Distortion parameters [num_params] as numpy array
        """
        transformed = self.transform(image=image_np)
        tensor = (
            torch.from_numpy(transformed["image"])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            _, _, params = self.model(tensor)

        return params.squeeze(0).cpu().numpy()

    def correct(
        self, image: Image.Image
    ) -> Tuple[Image.Image, Dict[str, float]]:
        """Run full cascade correction on a PIL image.

        Args:
            image: Input PIL image (any size)

        Returns:
            Tuple of (corrected PIL image, metadata dict with k1/k2/method info)
        """
        image_np = np.array(image.convert("RGB"))

        # Step 1: DL prediction
        params = self.predict_params(image_np)
        k1_dl = float(params[0])
        k2_dl = float(params[1]) if len(params) >= 2 else 0.0

        # Step 2: Classical refinement (uses DL k1 as starting point)
        k1_final, score, num_lines = optimize_k1(
            image_np, initial_k1=k1_dl, k2=k2_dl
        )

        if num_lines < self.min_lines:
            k1_final = k1_dl
            method = "dl_only"
        else:
            method = "cascade"

        k2_final = k2_dl

        # Step 3: Apply correction at full resolution
        corrected_np = apply_radial_correction(
            image_np, k1=k1_final, k2=k2_final
        )
        corrected = Image.fromarray(corrected_np)

        metadata = {
            "k1_dl": k1_dl,
            "k2_dl": k2_dl,
            "k1_classical": float(k1_final) if method == "cascade" else None,
            "k1_final": k1_final,
            "k2_final": k2_final,
            "num_lines": num_lines,
            "method": method,
            "straightness_score": score,
        }

        return corrected, metadata
