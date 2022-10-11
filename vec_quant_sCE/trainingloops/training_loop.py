import datetime
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
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
        self.patch_size = config["img_dims"]
        self.scales = config["scales"]
        self.EPOCHS = config["expt"]["epochs"]
        self.IMAGE_SAVE_PATH = f"{config['paths']['expt_path']}/images"
        self.MODEL_SAVE_PATH = f"{config['paths']['expt_path']}/models"
        self.LOG_SAVE_PATH = f"{config['paths']['expt_path']}/logs"
        self.SAVE_EVERY = config["expt"]["save_every"]

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/train"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/train")

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/validation"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/validation")

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.ds_train, self.ds_val = dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/train")
        self.test_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/test")

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
        self.Model.UNet.save_weights(f"{self.MODEL_SAVE_PATH}/model.ckpt")

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
                for scale in self.scales:
                    self._downsample_images(scale, **data)
                    for patch_data in self._sample_images(data["source"], data["target"]):
                        patch_data["times"] = data["times"]
                        self.Model.train_step(**patch_data)

            self._save_train_results(epoch)
            if verbose:
                print(f"Train epoch {epoch + 1}, L1, VQ, Total: {[metric.result().numpy() for metric in self.Model.metrics]}")

            # Validation step if appropriate
            if self.config["data"]["cv_folds"] > 1:
                self.Model.reset_metrics()

                # Run validation step for each batch in validation data
                for data in self.ds_val:
                    for scale in self.scales:
                        self._downsample_images(scale, **data)
                        for patch_data in self._sample_images(data["source"], data["target"]):
                            patch_data["times"] = data["times"]
                            self.Model.test_step(**patch_data)

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

        json.dump(self.results, open(f"{self.LOG_SAVE_PATH}/results.json", 'w'), indent=4)

    def _downsample_images(self, scale, source, target, seg=None, **kwargs):
        source = source[::scale, ::scale, :]
        target = target[::scale, ::scale, :]

        if seg is not None:
            seg = seg[::scale, ::scale, :]

    def _sample_images(self, source, target, seg=None):
        # Extract patches
        total_height = target.shape[0]
        total_width = target.shape[1]
        total_depth = target.shape[2]
        num_iter = (total_height // self.patch_size[0]) * (total_width // self.patch_size[1]) * (total_depth // self.patch_size[2])

        for _ in range(num_iter):
            x = np.random.randint(0, total_width - self.patch_size[0] + 1)
            y = np.random.randint(0, total_width - self.patch_size[1] + 1)
            z = np.random.randint(0, total_depth - self.patch_size[2] + 1)

            sub_target = target[:, x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), z:(z + self.patch_size[2]), :]
            sub_source = source[:, x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), z:(z + self.patch_size[2]), :]

            if seg is None:
                yield {
                "source": sub_source,
                "target": sub_target,
                "seg": None
                }

            else:
                sub_seg = seg[:, x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), z:(z + self.patch_size[2]), :]
                yield {
                "source": sub_source,
                "target": sub_target,
                "seg": sub_seg
                }

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
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{phase}/{epoch}.png", dpi=250)

        plt.close()
