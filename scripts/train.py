import argparse
import yaml
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch

from src.data.dataset import LensDistortionDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.training.trainer import LensCorrector

def main():
    parser = argparse.ArgumentParser(description="Train AutoHDR Lens Correction Model")
    parser.add_argument("--config", type=str, default="configs/train_fast.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint (path to .ckpt file)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    pl.seed_everything(config.get("seed", 42))

    # Image size setup
    img_size = tuple(config["model"].get("image_size", (512, 512)))

    # Dataset transforms
    train_transforms = get_train_transforms(size=img_size)
    val_transforms = get_val_transforms(size=img_size)

    # Data loaders
    data_dir = config["data"]["data_dir"]
    train_dir = Path(data_dir) / "train"

    # Training Dataset
    train_dataset = LensDistortionDataset(
        data_dir=train_dir,
        split="train",
        transform=train_transforms,
        val_split=config["data"]["val_split"],
        seed=config.get("seed", 42)
    )

    # Validation Dataset
    val_dataset = LensDistortionDataset(
        data_dir=train_dir,
        split="val",
        transform=val_transforms,
        val_split=config["data"]["val_split"],
        seed=config.get("seed", 42)
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Truncate dataset for fast runs if train_size is specified
    if "train_size" in config["data"]:
        train_size = config["data"]["train_size"]
        if len(train_dataset) > train_size:
            train_dataset.samples = train_dataset.samples[:train_size]
            print(f"Truncated training samples to: {len(train_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        drop_last=False
    )

    # Model
    model = LensCorrector(
        model_config=config["model"],
        loss_config=config["loss"],
        optim_config=config["training"]
    )

    # Callbacks
    cb_config = config.get("callbacks", {})
    callbacks = []

    if "model_checkpoint" in cb_config:
        callbacks.append(ModelCheckpoint(
            dirpath="outputs/models",
            monitor=cb_config["model_checkpoint"]["monitor"],
            mode=cb_config["model_checkpoint"]["mode"],
            save_top_k=cb_config["model_checkpoint"]["save_top_k"],
            filename=cb_config["model_checkpoint"]["filename"],
        ))

    if "early_stopping" in cb_config:
        callbacks.append(EarlyStopping(
            monitor=cb_config["early_stopping"]["monitor"],
            patience=cb_config["early_stopping"]["patience"],
            mode=cb_config["early_stopping"]["mode"],
            min_delta=cb_config["early_stopping"].get("min_delta", 0.0),
        ))

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(TQDMProgressBar(refresh_rate=1, leave=True))

    # Logger
    logger = TensorBoardLogger(
        save_dir=config["logging"]["save_dir"],
        name=config["logging"]["name"],
        version=config["logging"].get("version")
    )

    # Calculate accumulate grad batches
    accumulate_grad_batches = config["training"].get("accumulate_grad_batches", 1)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="auto",
        devices=1,
        precision=config["training"].get("precision", "32"),
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=config["training"].get("gradient_clip_val", 0.0),
        gradient_clip_algorithm=config["training"].get("gradient_clip_algorithm", "norm"),
        log_every_n_steps=config["logging"].get("log_every_n_steps", 10),
    )

    # Start Training (resume from checkpoint if provided)
    if args.ckpt:
        print(f"Resuming from checkpoint: {args.ckpt}")
    else:
        print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt,
    )
    print("Training finished!")

if __name__ == "__main__":
    main()