import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

np.set_printoptions(suppress=True)


class TrainingLoop:

    def __init__(self,
                 Model: object,
                 dataset: object,
                 train_generator: object,
                 val_generator: object,
                 config: dict):
        self.Model = Model
        self.config = config
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
        if self.config["expt"]["focal"]:
            self.results["train_L1"].append([float(self.Model.train_L1_metric.result()[0]), float(self.Model.train_L1_metric.result()[1])])

            if self.config["expt"]["log_scalars"]:
                with self.train_writer.as_default():
                    tf.summary.scalar("focal_L1", self.Model.train_L1_metric.result()[0], step=epoch)
                    tf.summary.scalar("global_L1", self.Model.train_L1_metric.result()[1], step=epoch)

        else:
            self.results["train_L1"].append(float(self.Model.train_L1_metric.result()))

            if self.config["expt"]["log_scalars"]:
                with self.train_writer.as_default():
                    tf.summary.scalar("L1", self.Model.train_L1_metric.result(), step=epoch)
        
        # Log parameter values
        if self.config["expt"]["log_histograms"]:
            with self.train_writer.as_default():
                for v in self.Model.UNet.trainable_variables:
                    tf.summary.histogram(v.name, v, step=epoch)

    def _save_val_results(self, epoch):
        # Log losses
        if self.config["expt"]["focal"]:
            self.results["val_L1"].append([float(self.Model.val_L1_metric.result()[0]), float(self.Model.val_L1_metric.result()[1])])

            if self.config["expt"]["log_scalars"]:
                with self.test_writer.as_default():
                    tf.summary.scalar("focal_L1", self.Model.val_L1_metric.result()[0], step=epoch)
                    tf.summary.scalar("global_L1", self.Model.val_L1_metric.result()[1], step=epoch)

        else:
            self.results["val_L1"].append(float(self.Model.val_L1_metric.result()))

            if self.config["expt"]["log_scalars"]:
                with self.test_writer.as_default():
                    tf.summary.scalar("L1", self.Model.val_L1_metric.result(), step=epoch)

    def _process_and_save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()

        if "target_times" in data.keys():
            pred = self.Model.UNet(data["real_source"], data["target_times"]).numpy()
        else:
            pred = self.Model.UNet(data["real_source"]).numpy()

        source = data_generator.un_normalise(data["real_source"])
        target = data_generator.un_normalise(data["real_target"])
        pred = data_generator.un_normalise(pred)

        self._save_images(epoch, phase, tuning_path, source, target, pred)

    def _save_model(self):
        self.Model.UNet.save_weights(f"{self.MODEL_SAVE_PATH}/model.ckpt")


    def _save_images(self, epoch, phase, tuning_path, source, target, pred):
        fig, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(np.abs(target[i, :, :, 11, 0] - source[i, :, :, 11, 0]), cmap="hot")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(np.abs(target[i, :, :, 11, 0] - pred[i, :, :, 11, 0]), cmap="hot")
            axs[i, 4].axis("off")

        plt.tight_layout()

        if tuning_path:
            plt.savefig(f"{tuning_path}.png", dpi=250)
        else:
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{phase}/{epoch}.png", dpi=250)

        plt.close()

    def train(self, verbose=1):

        """ Main training loop for U-Net """

        self.results = {}
        self.results["train_L1"] = []
        self.results["val_L1"] = []
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()

        for epoch in range(self.EPOCHS):
            self.Model.reset_train_metrics()

            # Run training step for each batch in training data
            for data in self.ds_train:
                self.Model.train_step(**data)

            self._save_train_results(epoch)
            if verbose:
                self._print_results(epoch=epoch, training=True)

            # Validation step if appropriate
            if self.config["data"]["cv_folds"] > 1:
                self.Model.val_L1_metric.reset_states()

                # Run validation step for each batch in validation data
                for data in self.ds_val:
                    self.Model.test_step(**data)

                self._save_val_results(epoch)
                if verbose:
                    self._print_results(epoch=epoch, training=False)

            # Save example images
            if (epoch + 1) % self.SAVE_EVERY == 0:
                self._process_and_save_images(epoch + 1, phase="train")
                self._process_and_save_images(epoch + 1, phase="validation")

            # Save model if necessary
            if (epoch + 1) % self.SAVE_EVERY == 0 and self.config["expt"]["save_model"]:
                self._save_model()

        self.results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")

        json.dump(self.results, open(f"{self.LOG_SAVE_PATH}/results.json", 'w'), indent=4)
    
    def save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()

        if self.config["expt"]["model"] == "HyperPix2Pix" and "source_times" in data.keys():
            pred = np.zeros_like(data["real_source"])

            for i in range(data["real_source"].shape[0]):
                pred[i, ...] = self.Model.Generator(data["real_source"][i, ...][tf.newaxis, :, :, :, :], data["target_times"][i][tf.newaxis]).numpy()

        elif self.config["expt"]["model"] == "HyperPix2Pix" and "source_times" not in data.keys():
            pred = np.zeros_like(data["real_source"])

            for i in range(data["real_source"].shape[0]):
                pred[i, ...] = self.Model.Generator(data["real_source"][i, ...][tf.newaxis, :, :, :, :]).numpy()

        elif self.config["expt"]["model"] == "Pix2Pix" and "source_times" in data.keys():
            pred = self.Model.Generator(data["real_source"], data["target_times"]).numpy()

        else:
            pred = self.Model.Generator(data["real_source"]).numpy()

        source = data_generator.un_normalise(data["real_source"])
        target = data_generator.un_normalise(data["real_target"])
        pred = data_generator.un_normalise(pred)

        fig, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(np.abs(target[i, :, :, 11, 0] - source[i, :, :, 11, 0]), cmap="hot")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(np.abs(target[i, :, :, 11, 0] - pred[i, :, :, 11, 0]), cmap="hot")
            axs[i, 4].axis("off")

        plt.tight_layout()

        if tuning_path:
            plt.savefig(f"{tuning_path}.png", dpi=250)
        else:
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{phase}/{epoch}.png", dpi=250)

        plt.close()
