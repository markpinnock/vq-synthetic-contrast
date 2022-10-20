import datetime
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import time

np.set_printoptions(precision=4, suppress=True)


class TrainingLoop:

    def __init__(self,
                 Model: object,
                 dataset: object,
                 train_generator: object,
                 val_generator: object,
                 config: dict):
        self.Model = Model
        self.config = config
        self.intermediate_vq = "output" in config["hyperparameters"]["vq_layers"]
        self.patch_size = config["hyperparameters"]["img_dims"]
        self.scales = config["hyperparameters"]["scales"]
        self.img_dims = config["hyperparameters"]["img_dims"]
        self.EPOCHS = config["expt"]["epochs"]

        expt_path = Path(config["paths"]["expt_path"])
        self.image_save_path = expt_path / "images"
        self.image_save_path.mkdir(parents=True, exist_ok=True)
        self.image_save_path / "train"
        self.model_save_path = expt_path / "models"
        self.log_save_path = expt_path / "logs"
        self.SAVE_EVERY = config["expt"]["save_every"]

        if not os.path.exists(self.image_save_path / "train"):
            os.mkdir(self.image_save_path / "train")

        if not os.path.exists(self.image_save_path / "validation"):
            os.mkdir(self.image_save_path / "validation")

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.ds_train, self.ds_val = dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(str(self.log_save_path / log_time / "train"))
        self.test_writer = tf.summary.create_file_writer(str(self.log_save_path / log_time / "test"))

    def _save_train_results(self, epoch):
        # Log losses
        self.results["train_L1"].append(float(self.Model.L1_metric.result()))
        self.results["train_vq"].append(float(self.Model.vq_metric.result()))
        self.results["train_total"].append(float(self.Model.total_metric.result()))

        if self.config["expt"]["log_scalars"]:
            with self.train_writer.as_default():
                tf.summary.scalar("train_L1", self.Model.L1_metric.result(), step=epoch)
                tf.summary.scalar("train_vq", self.Model.vq_metric.result(), step=epoch)
                tf.summary.scalar("train_total", self.Model.total_metric.result(), step=epoch)
        
        # Log parameter values
        if self.config["expt"]["log_histograms"]:
            with self.train_writer.as_default():
                for v in self.Model.UNet.trainable_variables:
                    tf.summary.histogram(v.name, v, step=epoch)

    def _save_val_results(self, epoch):
        # Log losses
        self.results["val_L1"].append(float(self.Model.L1_metric.result()))
        self.results["val_vq"].append(float(self.Model.vq_metric.result()))
        self.results["val_total"].append(float(self.Model.total_metric.result()))

        if self.config["expt"]["log_scalars"]:
            with self.test_writer.as_default():
                tf.summary.scalar("val_L1", self.Model.L1_metric.result(), step=epoch)
                tf.summary.scalar("val_vq", self.Model.vq_metric.result(), step=epoch)
                tf.summary.scalar("val_total", self.Model.total_metric.result(), step=epoch)

    def _save_model(self):
        self.Model.UNet.save_weights(self.model_save_path / "model.ckpt")

    def train(self, verbose=1):

        """ Main training loop for U-Net """

        self.results = {}
        self.results["train_L1"] = []
        self.results["train_vq"] = []
        self.results["train_total"] = []
        self.results["val_L1"] = []
        self.results["val_vq"] = []
        self.results["val_total"] = []
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()

        for epoch in range(self.EPOCHS):
            self.Model.reset_metrics()

            # Run training step for each batch in training data
            for data in self.ds_train:
                source = data["source"]
                target = data["target"]
                seg = data["seg"] if "seg" in data.keys() else None

                # Augmentation if required
                # if self.Model.Aug:
                #     (source, target), seg = self.Model.Aug(imgs=[source, target], seg=seg)
                plt.subplot(2, 2, 1)
                plt.imshow(source[0, :, :, 0, 0])
                plt.subplot(2, 2, 2)
                plt.imshow(target[0, :, :, 0, 0])
                plt.subplot(2, 2, 3)
                plt.imshow(source[1, :, :, 0, 0])
                plt.subplot(2, 2, 4)
                plt.imshow(target[1, :, :, 0, 0])
                plt.show()
                source = self._downsample_images(source)
                plt.subplot(2, 2, 1)
                plt.imshow(source[0, :, :, 0, 0])
                plt.subplot(2, 2, 2)
                plt.imshow(target[0, :, :, 0, 0])
                plt.subplot(2, 2, 3)
                plt.imshow(source[1, :, :, 0, 0])
                plt.subplot(2, 2, 4)
                plt.imshow(target[1, :, :, 0, 0])
                plt.show()
                # Randomise segments of image to sample, iteratively get random indices of smaller segments
                x, y = self._get_scale_indices()
                print(x, y)
                target, seg = self._sample_patches(x[-1], y[-1], target, seg)
                plt.subplot(2, 2, 1)
                plt.imshow(source[0, :, :, 0, 0])
                plt.subplot(2, 2, 2)
                plt.imshow(target[0, :, :, 0, 0])
                plt.subplot(2, 2, 3)
                plt.imshow(source[1, :, :, 0, 0])
                plt.subplot(2, 2, 4)
                plt.imshow(target[1, :, :, 0, 0])
                plt.show()
                # Perform multi-scale training
                source, _ = self._sample_patches(x[0], y[0], source)
                pred, vq = self.Model(source, data["times"])
                print(source.shape, pred.shape, target.shape, "!")

                plt.subplot(2, 3, 1)
                plt.imshow(source[0, :, :, 0, 0])
                plt.subplot(2, 3, 2)
                plt.imshow(pred[0, :, :, 0, 0])
                plt.subplot(2, 3, 3)
                plt.imshow(target[0, :, :, 0, 0])
                plt.subplot(2, 3, 4)
                plt.imshow(source[1, :, :, 0, 0])
                plt.subplot(2, 3, 5)
                plt.imshow(pred[1, :, :, 0, 0])
                plt.subplot(2, 3, 6)
                plt.imshow(target[1, :, :, 0, 0])
                plt.show()
                for i in range(1, len(self.scales)):
                    scale_factor = 2 ** i
                    if (x[i] / scale_factor != x[i] // scale_factor) or (y[i] / scale_factor != y[i] // scale_factor):
                        raise ValueError
                    pred, _ = self._sample_patches(x[i] // scale_factor, y[i] // scale_factor, pred)
                    print(source.shape, pred.shape, target.shape, "?", x[i] // scale_factor, y[i] // scale_factor)
                    plt.subplot(2, 3, 1)
                    plt.imshow(source[0, :, :, 0, 0])
                    plt.subplot(2, 3, 2)
                    plt.imshow(pred[0, :, :, 0, 0])
                    plt.subplot(2, 3, 3)
                    plt.imshow(target[0, :, :, 0, 0])
                    plt.subplot(2, 3, 4)
                    plt.imshow(source[1, :, :, 0, 0])
                    plt.subplot(2, 3, 5)
                    plt.imshow(pred[1, :, :, 0, 0])
                    plt.subplot(2, 3, 6)
                    plt.imshow(target[1, :, :, 0, 0])
                    plt.show()
                    if self.intermediate_vq:
                        pred, vq = self.Model(vq, data["times"])
                    else:
                        pred, vq = self.Model(pred, data["times"])
                    print(source.shape, pred.shape, target.shape, "#")
                    plt.subplot(2, 3, 1)
                    plt.imshow(source[0, :, :, 0, 0])
                    plt.subplot(2, 3, 2)
                    plt.imshow(pred[0, :, :, 0, 0])
                    plt.subplot(2, 3, 3)
                    plt.imshow(target[0, :, :, 0, 0])
                    plt.subplot(2, 3, 4)
                    plt.imshow(source[1, :, :, 0, 0])
                    plt.subplot(2, 3, 5)
                    plt.imshow(pred[1, :, :, 0, 0])
                    plt.subplot(2, 3, 6)
                    plt.imshow(target[1, :, :, 0, 0])
                    plt.show()
                plt.subplot(2, 3, 1)
                plt.imshow(source[0, :, :, 0, 0])
                plt.subplot(2, 3, 2)
                plt.imshow(pred[0, :, :, 0, 0])
                plt.subplot(2, 3, 3)
                plt.imshow(target[0, :, :, 0, 0])
                plt.subplot(2, 3, 4)
                plt.imshow(source[1, :, :, 0, 0])
                plt.subplot(2, 3, 5)
                plt.imshow(pred[1, :, :, 0, 0])
                plt.subplot(2, 3, 6)
                plt.imshow(target[1, :, :, 0, 0])
                plt.show()
                self.Model.train_step(pred, target, seg, data["times"])

            self._save_train_results(epoch)
            if verbose:
                print(f"Train epoch {epoch + 1}, L1, VQ, Total: {[metric.result().numpy() for metric in self.Model.metrics]}")

            # Validation step if appropriate
            if self.config["data"]["cv_folds"] > 1:
                self.Model.reset_metrics()

                # Run validation step for each batch in validation data
                for data in self.ds_val:
                    source = self._downsample_images(data["source"])

                    # Randomise segments of image to sample, iteratively get random indices of smaller segments
                    x, y = self._get_scale_indices()
                    target, seg = self._sample_patches(x[-1], y[-1], target, seg)

                    # Perform multi-scale inference
                    source = self._sample_patches(x[0], y[0], source)
                    pred, vq = self.Model(source, data["times"])
                    for i in range(1, len(self.scales)):
                        pred = self._sample_patches(x[i], y[i], pred, vq)
                        if self.intermediate_vq:
                            pred, vq = self.Model(vq, data["times"])
                        else:
                            pred, vq = self.Model(pred, data["times"])

                    self.Model.test_step(pred, target, seg, data["times"])

                self._save_val_results(epoch)
                if verbose:
                    print(f"Val epoch {epoch + 1}, L1, VQ, Total: {[metric.result().numpy() for metric in self.Model.metrics]}")

            # Save example images
            if (epoch + 1) % self.SAVE_EVERY == 0:
                self._save_images(epoch + 1, phase="train")
                self._save_images(epoch + 1, phase="validation")

            # Save model if necessary
            if (epoch + 1) % self.SAVE_EVERY == 0 and self.config["expt"]["save_model"]:
                self._save_model()

        self.results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")

        json.dump(self.results, open(f"{self.log_save_path}/results.json", 'w'), indent=4)

    def _get_scale_indices(self):
        xs, ys = [], []
        x = np.random.randint(0, self.scales[-1])
        y = np.random.randint(0, self.scales[-1])
        xs.append(x * self.img_dims[0])
        ys.append(y * self.img_dims[1])

        for i in range(len(self.scales) - 2, -1, -1):
            scale_factor = self.scales[i] // self.scales[i + 1]
            x = np.random.randint(x * scale_factor, x * scale_factor + 2)
            y = np.random.randint(y * scale_factor, y * scale_factor + 2)
            xs.append(x * self.img_dims[0])
            ys.append(y * self.img_dims[1])

        return xs, ys

    def _downsample_images(self, img):
        img = img[:, ::self.scales[0], ::self.scales[0], :, :]
        return img

    def _sample_patches(self, x, y, img1, img2=None):

        img1 = img1[:, x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), :, :]

        if img2 is not None:
            img2 = img2[:, x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), :, :]

        return img1, img2

    def _save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()

        if "times" in data.keys():
            pred = self.Model(data["source"], data["times"]).numpy()
        else:
            pred = self.Model(data["source"], None).numpy()

        source = data_generator.un_normalise(data["source"])
        target = data_generator.un_normalise(data["target"])
        pred = data_generator.un_normalise(pred)

        fig, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(target[i, :, :, 11, 0] - source[i, :, :, 11, 0], norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(np.abs(target[i, :, :, 11, 0] - pred[i, :, :, 11, 0]), norm=mpl.colors.CenteredNorm(), cmap="bwr")
            axs[i, 4].axis("off")

        plt.tight_layout()

        if tuning_path:
            plt.savefig(f"{tuning_path}.png", dpi=250)
        else:
            plt.savefig(self.image_save_path / phase / epoch / ".png", dpi=250)

        plt.close()
