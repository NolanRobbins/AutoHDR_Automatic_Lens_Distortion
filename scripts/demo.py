import argparse
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from src.data.transforms import get_test_transforms
from src.inference.ensemble import EnsembleLensCorrector
from src.training.trainer import LensCorrector


def get_demo_examples() -> List[List[str]]:
    """Build example list: include tougher (high-distortion) images to showcase DL correction."""
    base = Path("data/test")
    if not base.exists():
        return [
            ["data/test/008195cc-eadd-42bb-99b9-108deb738154_g0.jpg"],
            ["data/test/008195cc-eadd-42bb-99b9-108deb738154_g10.jpg"],
        ]
    # Prefer higher _g* = more distortion (tougher); add variety of scene IDs
    candidates = sorted(base.glob("*.jpg"))
    by_id: dict[str, list[Path]] = {}
    for p in candidates:
        stem = p.stem
        if "_g" in stem:
            scene_id, g_part = stem.rsplit("_g", 1)
            try:
                g = int(g_part)
                by_id.setdefault(scene_id, []).append((g, p))
            except ValueError:
                by_id.setdefault(stem, []).append((0, p))
        else:
            by_id.setdefault(stem, []).append((0, p))
    examples = []
    seen = set()
    # Add high-distortion first (g10, g15, g8, etc.) then medium/low
    for scene_id, pairs in by_id.items():
        pairs.sort(key=lambda x: -x[0])  # descending g
        for g, p in pairs[:2]:
            path = str(p)
            if path not in seen:
                seen.add(path)
                examples.append([path])
    # Fallback if no test images found
    if not examples:
        examples = [
            ["data/test/008195cc-eadd-42bb-99b9-108deb738154_g0.jpg"],
            ["data/test/008195cc-eadd-42bb-99b9-108deb738154_g10.jpg"],
        ]
    return examples[:8]


def load_model(checkpoint_path: str, config_path: str):
    """Load the trained PyTorch Lightning model from a checkpoint."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = LensCorrector.load_from_checkpoint(
        checkpoint_path,
        model_config=config["model"],
        loss_config=config["loss"],
        optim_config=config["training"],
        map_location="cpu",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, config, device


def process_image_dl(model, config, device, input_image: Image.Image) -> Image.Image:
    """Run DL-only inference on a single image."""
    if input_image is None:
        return None

    img_size = tuple(config["model"].get("image_size", (224, 224)))
    orig_w, orig_h = input_image.size

    transform = get_test_transforms(size=img_size)
    img_np = np.array(input_image.convert("RGB"))
    transformed = transform(image=img_np)
    img_tensor = (
        torch.from_numpy(transformed["image"])
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(img_tensor)
        corrected_tensor = output[0] if isinstance(output, tuple) else output

    corrected_tensor = corrected_tensor.squeeze(0).cpu()
    if corrected_tensor.shape[0] > 3:
        corrected_tensor = corrected_tensor[:3, :, :]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    corrected_tensor = torch.clamp(corrected_tensor * std + mean, 0, 1)

    corrected_img = TF.to_pil_image(corrected_tensor)
    corrected_img = corrected_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    return corrected_img


def main():
    parser = argparse.ArgumentParser(description="AutoHDR Lens Correction Demo")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_radial.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["dl", "ensemble"],
        default="ensemble",
        help="dl = DL model only, ensemble = DL + classical refinement (default)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server (default 7860)")
    args = parser.parse_args()

    if args.mode == "ensemble":
        print(f"Loading ensemble (DL + classical) from {args.ckpt}...")
        corrector = EnsembleLensCorrector(args.ckpt, args.config)
        print(f"Ensemble ready on {corrector.device}!")

        def predict(image):
            if image is None:
                return None, ""
            corrected, meta = corrector.correct(image)
            info = (
                f"Method: {meta['method']}  |  "
                f"k1: {meta['k1_final']:.4f}  |  k2: {meta['k2_final']:.4f}  |  "
                f"Lines detected: {meta['num_lines']}  |  "
                f"DL k1: {meta['k1_dl']:.4f}"
            )
            return corrected, info
    else:
        print(f"Loading DL model from {args.ckpt}...")
        model, config, device = load_model(args.ckpt, args.config)
        print(f"Model loaded on {device}!")

        def predict(image):
            if image is None:
                return None, ""
            corrected = process_image_dl(model, config, device, image)
            return corrected, "Mode: DL only"

    with gr.Blocks(title="AutoHDR Lens Correction") as demo:
        gr.Markdown(
            """
            # AutoHDR Lens Distortion Correction
            Upload a distorted real-estate photo to automatically correct barrel and pincushion distortion.
            Uses a **Swin Transformer** to predict radial distortion parameters (k1/k2), optionally
            refined by classical line-straightness optimization.
            **Examples below** include tougher, high-distortion images to showcase the deep-learning correction.
            """
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Original Distorted Image")
                submit_btn = gr.Button("Correct Lens Distortion", variant="primary")

            with gr.Column():
                output_img = gr.Image(type="pil", label="Corrected Image", interactive=False)
                info_text = gr.Textbox(label="Correction Details", interactive=False)

        submit_btn.click(fn=predict, inputs=input_img, outputs=[output_img, info_text])

        gr.Examples(
            examples=get_demo_examples(),
            inputs=input_img,
            label="Example images (including high-distortion to showcase DL)",
        )

    demo.launch(share=False, server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()