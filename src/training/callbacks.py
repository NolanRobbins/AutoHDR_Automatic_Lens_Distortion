"""Custom PyTorch Lightning callbacks."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LogPredictionSamplesCallback(Callback):
    """
    Callback to log prediction samples during validation.

    Logs a few example predictions to WandB/TensorBoard for visual inspection.
    """

    def __init__(self, num_samples: int = 4) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log predictions at end of validation epoch."""
        # TODO: Implement image logging
        pass
