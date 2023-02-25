import datetime
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import time

from vq_sce import ABDO_WINDOW, LQ_DEPTH

np.set_printoptions(precision=4, suppress=True)


class TrainingLoop:

    def __init__(self,
                 Model: object,
                 train_dataset: object,
                 val_dataset: object,
                 train_generator: object,
                 val_generator: object,
                 config: dict):
        self.Model = Model
        self.config = config
        self.epochs = config["expt"]["epochs"]

        expt_path = Path(config["paths"]["expt_path"])
        self.image_save_path = expt_path / "images"
        self.image_save_path.mkdir(parents=True, exist_ok=True)
        self.image_save_path / "train"
        self.model_save_path = expt_path / "models"
        self.log_save_path = expt_path / "logs"
        self.save_every = config["expt"]["save_every"]

        if "scales" not in config["hyperparameters"].keys():
            self.multi_scale = False
        elif len(config["hyperparameters"]["scales"]) == 1:
            self.multi_scale = False
        else:
            self.multi_scale = True

        if not os.path.exists(self.image_save_path / "train"):
            os.mkdir(self.image_save_path / "train")

        if not os.path.exists(self.image_save_path / "validation"):
            os.mkdir(self.image_save_path / "validation")

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.ds_train, self.ds_val = train_dataset, val_dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(str(self.log_save_path / log_time / "train"))
        self.test_writer = tf.summary.create_file_writer(str(self.log_save_path / log_time / "test"))

    def _save_train_results(self, epoch):
        # Log losses
        prefix = "ce" if self.config["data"]["type"] == "contrast" else "sr"
        self.results[f"train_{prefix}_L1"].append(float(self.Model.L1_metric.result()))
        self.results[f"train_{prefix}_vq"].append(float(self.Model.vq_metric.result()))
        self.results[f"train_{prefix}_total"].append(float(self.Model.total_metric.result()))

        if self.config["expt"]["log_scalars"]:
            with self.train_writer.as_default():
                tf.summary.scalar(f"train_{prefix}_L1", self.Model.L1_metric.result(), step=epoch)
                tf.summary.scalar(f"train_{prefix}_vq", self.Model.vq_metric.result(), step=epoch)
                tf.summary.scalar(f"train_{prefix}_total", self.Model.total_metric.result(), step=epoch)
        
        # Log parameter values
        if self.config["expt"]["log_histograms"]:
            with self.train_writer.as_default():
                for v in self.Model.UNet.trainable_variables:
                    tf.summary.histogram(v.name, v, step=epoch)

    def _save_val_results(self, epoch):
        # Log losses
        prefix = "ce" if self.config["data"]["type"] == "contrast" else "sr"
        self.results[f"val_{prefix}_L1"].append(float(self.Model.L1_metric.result()))
        self.results[f"val_{prefix}_vq"].append(float(self.Model.vq_metric.result()))
        self.results[f"val_{prefix}_total"].append(float(self.Model.total_metric.result()))

        if self.config["expt"]["log_scalars"]:
            with self.test_writer.as_default():
                tf.summary.scalar(f"val_{prefix}_L1", self.Model.L1_metric.result(), step=epoch)
                tf.summary.scalar(f"val_{prefix}_vq", self.Model.vq_metric.result(), step=epoch)
                tf.summary.scalar(f"val_{prefix}_total", self.Model.total_metric.result(), step=epoch)

    def _save_model(self):
        self.Model.UNet.save_weights(self.model_save_path / "model.ckpt")

    def train(self, verbose=1):

        """ Main training loop for U-Net """

        prefix = "ce" if self.config["data"]["type"] == "contrast" else "sr"
        self.results = {}
        self.results[f"train_{prefix}_L1"] = []
        self.results[f"train_{prefix}_vq"] = []
        self.results[f"train_{prefix}_total"] = []
        self.results[f"val_{prefix}_L1"] = []
        self.results[f"val_{prefix}_vq"] = []
        self.results[f"val_{prefix}_total"] = []
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            self.Model.reset_metrics()

            # Run training step for each batch in training data
            for data in self.ds_train:
                self.Model.train_step(**data)

            self._save_train_results(epoch)
            if verbose:
                print(f"Train epoch {epoch + 1}, L1, VQ, Total: {[metric.result().numpy() for metric in self.Model.metrics]}")

            # Validation step if appropriate
            if self.config["data"]["cv_folds"] > 1:
                self.Model.reset_metrics()

                # Run validation step for each batch in validation data
                for data in self.ds_val:
                    self.Model.test_step(**data)

                self._save_val_results(epoch)
                if verbose:
                    print(f"Val epoch {epoch + 1}, L1, VQ, Total: {[metric.result().numpy() for metric in self.Model.metrics]}")

            # Save example images
            if (epoch + 1) % self.save_every == 0:
                if self.multi_scale:
                    self._save_multiscale_images(epoch + 1, phase="train")
                    self._save_multiscale_images(epoch + 1, phase="validation")
                else:
                    self._save_images(epoch + 1, phase="train")
                    self._save_images(epoch + 1, phase="validation")

            # Save results
            json.dump(self.results, open(f"{self.log_save_path}/results.json", 'w'), indent=4)

            # Save model if necessary
            if (epoch + 1) % self.save_every == 0 and self.config["expt"]["save_model"]:
                self._save_model()

        self.results["time"] = (time.time() - start_time) / 3600
        json.dump(self.results, open(f"{self.log_save_path}/results.json", 'w'), indent=4)
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")

    def _save_images(self, epoch, phase="validation"):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images
        source, target, pred = self.Model.example_inference(**data)

        source = data_generator.un_normalise(source)
        target = data_generator.un_normalise(target)
        pred = data_generator.un_normalise(pred)

        source_mid = 1 if source.shape[1] == LQ_DEPTH else 5
        target_mid = 5

        _, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, source_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(target[i, target_mid, :, :, 0] - source[i, source_mid, :, :, 0], norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(target[i, target_mid, :, :, 0] - pred[i, target_mid, :, :, 0], norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 4].axis("off")

        plt.tight_layout()
        plt.savefig(self.image_save_path / phase / f"{epoch}.png", dpi=250)
        plt.close()

    def _save_multiscale_images(self, epoch, phase="validation"):

        """ Saves sample of images from multi-scale U-Net """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images
        source, target, pred = self.Model.example_inference(**data)

        source = data_generator.un_normalise(source)
        target = data_generator.un_normalise(target)

        source_mid = 1 if source.shape[1] == LQ_DEPTH else 5
        target_mid = 5

        for scale in pred.keys():
            pred[scale] = data_generator.un_normalise(pred[scale])

        _, axs = plt.subplots(target.shape[0], 4 + len(pred.keys()))

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, source_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 0].axis("off")
            for j, img in enumerate(pred.values()):
                axs[i, 1 + j].imshow(img[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
                axs[i, 1 + j].axis("off")
            axs[i, 2 + j].imshow(target[i, target_mid, :, :, 0], cmap="bone", **ABDO_WINDOW)
            axs[i, 2 + j].axis("off")
            axs[i, 3 + j].imshow(target[i, target_mid, :, :, 0] - source[i, source_mid, :, :, 0], norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 3 + j].axis("off")
            axs[i, 4 + j].imshow(target[i, target_mid, :, :, 0] - list(pred.values())[-1][i, target_mid, :, :, 0], norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 4 + j].axis("off")

        plt.tight_layout()
        plt.savefig(self.image_save_path / phase / f"{epoch}.png", dpi=250)
        plt.close()
