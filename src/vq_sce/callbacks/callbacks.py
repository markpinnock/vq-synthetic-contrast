import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from vq_sce import ABDO_WINDOW, LQ_DEPTH
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader

# -------------------------------------------------------------------------


class SaveResults(tf.keras.callbacks.Callback):
    """Save metrics to json every N epochs."""

    def __init__(
        self,
        filepath: Path,
        save_freq: int,
        data_type: str,
        expt_type: str,
        opt_type: str,
    ) -> None:
        super().__init__()
        self.log_path = filepath
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

        try:
            with open(self.log_path / "results.json") as fp:
                self.results = json.load(fp)

        except FileNotFoundError:
            if expt_type == Task.JOINT:
                self.results = {
                    "train_sr_L1": [],
                    "valid_sr_L1": [],
                    "train_sr_vq": [],
                    "valid_sr_vq": [],
                    "train_ce_L1": [],
                    "valid_ce_L1": [],
                    "train_ce_vq": [],
                    "valid_ce_vq": [],
                }
                if opt_type == "DARTS":
                    self.results["train_alpha"] = []

            else:
                prefix = "ce" if data_type == Task.CONTRAST else "sr"
                self.results = {
                    f"train_{prefix}_L1": [],
                    f"valid_{prefix}_L1": [],
                    f"train_{prefix}_vq": [],
                    f"valid_{prefix}_vq": [],
                }

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Save results."""
        for metric_name, metric in logs.items():
            if "val" in metric_name:
                self.results[f"valid_{metric_name.strip('val_')}"].append(metric)
            else:
                self.results[f"train_{metric_name}"].append(metric)

        if (epoch + 1) % self.save_freq == 0:
            with open(self.log_path / "results.json", "w") as fp:
                json.dump(self.results, fp, indent=4)


# -------------------------------------------------------------------------


class SaveModel(tf.keras.callbacks.Callback):
    """Save model every N epochs."""

    def __init__(
        self,
        filepath: Path,
        save_freq: int,
        save_weights_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = filepath
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Save results."""
        if (epoch + 1) % self.save_freq == 0:
            if self.save_weights_only:
                self.model.save_weights(
                    self.model_path / f"ckpt-{epoch + 1}",
                    overwrite=True,
                )
            else:
                self.model.save(self.model_path / f"ckpt-{epoch + 1}", overwrite=True)


# -------------------------------------------------------------------------


class SaveExamples(tf.keras.callbacks.Callback):
    """Save example predictions from U-Net every N epochs."""

    def __init__(
        self,
        filepath: Path,
        save_freq: int,
        train_generator: BaseDataloader,
        valid_generator: BaseDataloader,
    ) -> None:
        super().__init__()
        self.image_path = filepath
        (self.image_path / "train").mkdir(parents=True, exist_ok=True)
        (self.image_path / "validation").mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

        self.train_generator = train_generator
        self.valid_generator = valid_generator

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Save example predictions."""
        if (epoch + 1) % self.save_freq == 0:
            source, target, pred = self._generate_predictions(self.train_generator)
            self._save_images("train", epoch, source, target, pred)
            source, target, pred = self._generate_predictions(self.valid_generator)
            self._save_images("validation", epoch, source, target, pred)

    def _generate_predictions(
        self,
        generator: BaseDataloader,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Generate example output."""
        data = generator.example_images
        source, target, pred = self.model.example_inference(**data)

        source = generator.un_normalise(source)
        target = generator.un_normalise(target)
        pred = generator.un_normalise(pred)

        return source, target, pred

    def _save_images(
        self,
        phase: str,
        epoch: int,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        pred: npt.NDArray[np.float32],
    ) -> None:
        """Save images.
        :param phase: `train` or `validation`
        """
        source_mid = 1 if source.shape[1] == LQ_DEPTH else 5
        target_mid = 5

        _, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, source_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(
                target[i, target_mid, :, :, 0] - source[i, source_mid, :, :, 0],
                norm=mpl.colors.CenteredNorm(),
                cmap="bwr",
            )
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(
                target[i, target_mid, :, :, 0] - pred[i, target_mid, :, :, 0],
                norm=mpl.colors.CenteredNorm(),
                cmap="bwr",
            )
            axs[i, 4].axis("off")

        plt.tight_layout()
        plt.savefig(self.image_path / phase / f"{epoch + 1}.png", dpi=250)
        plt.close()


# -------------------------------------------------------------------------


class SaveMultiScaleExamples(tf.keras.callbacks.Callback):
    """Save example predictions from multi-scale U-Net every N epochs."""

    def __init__(
        self,
        filepath: Path,
        save_freq: int,
        train_generator: BaseDataloader,
        valid_generator: BaseDataloader,
    ):
        super().__init__()
        self.image_path = filepath
        (self.image_path / "train").mkdir(parents=True, exist_ok=True)
        (self.image_path / "validation").mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

        self.train_generator = train_generator
        self.valid_generator = valid_generator

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Save example predictions."""
        if (epoch + 1) % self.save_freq == 0:
            source, target, pred = self._generate_predictions(self.train_generator)
            self._save_images("train", epoch, source, target, pred)
            source, target, pred = self._generate_predictions(self.valid_generator)
            self._save_images("validation", epoch, source, target, pred)

    def _generate_predictions(
        self,
        generator: BaseDataloader,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        dict[str, npt.NDArray[np.float32]],
    ]:
        """Generate example output."""
        data = generator.example_images
        source, target, pred = self.model.example_inference(**data)

        source = generator.un_normalise(source)
        target = generator.un_normalise(target)
        for scale in pred.keys():
            pred[scale] = self.train_generator.un_normalise(pred[scale])

        return source, target, pred

    def _save_images(
        self,
        phase: str,
        epoch: int,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        pred: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        """Save images.
        :param phase: `train` or `validation`
        """
        source_mid = 1 if source.shape[1] == LQ_DEPTH else 5
        target_mid = 5

        _, axs = plt.subplots(target.shape[0], 4 + len(pred.keys()))

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, source_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 0].axis("off")
            for j, img in enumerate(pred.values()):
                axs[i, 1 + j].imshow(
                    img[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW
                )
                axs[i, 1 + j].axis("off")
            axs[i, 2 + j].imshow(
                target[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW
            )
            axs[i, 2 + j].axis("off")
            axs[i, 3 + j].imshow(
                target[i, target_mid, :, :, 0] - source[i, source_mid, :, :, 0],
                norm=mpl.colors.CenteredNorm(),
                cmap="bwr",
            )
            axs[i, 3 + j].axis("off")
            axs[i, 4 + j].imshow(
                target[i, target_mid, :, :, 0]
                - list(pred.values())[-1][i, target_mid, :, :, 0],
                norm=mpl.colors.CenteredNorm(),
                cmap="bwr",
            )
            axs[i, 4 + j].axis("off")

        plt.tight_layout()
        plt.savefig(self.image_path / phase / f"{epoch + 1}.png", dpi=250)
        plt.close()
