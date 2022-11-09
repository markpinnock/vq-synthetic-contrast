import numpy as np
import tensorflow as tf

from .components.unet import UNet
from vq_sce.utils.augmentation import StdAug
from vq_sce.utils.losses import L1, FocalLoss


#-------------------------------------------------------------------------
""" Wrapper for multi-scale U-Net """

class Model(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.mb_size = config["expt"]["mb_size"]
        self.img_dims = config["data"]["patch_size"]
        config["hyperparameters"]["img_dims"] = self.img_dims
        self.intermediate_vq = "output" in config["hyperparameters"]["vq_layers"]
        self.scales = [
            config["augmentation"]["img_dims"][0] // config["data"]["patch_size"][0]
        ]

        if config["hyperparameters"]["time_layers"] is None:
            self.input_times = False
        else:
            self.input_times = True

        if config["hyperparameters"]["vq_layers"] is None:
            self.use_vq = False
        else:
            self.use_vq = True

        # Set up augmentation
        aug_config = config["augmentation"]
        aug_config["segs"] = config["data"]["segs"]

        if config["augmentation"]["use"]:
            self.Aug = StdAug(config=aug_config)
        else:
            self.Aug = None

        self.UNet = UNet(self.initialiser, config["hyperparameters"], name="unet")

    def compile(self, optimiser):
        self.optimiser = optimiser

        if self.config["expt"]["focal"]:
            self.L1_loss = FocalLoss(self.config["hyperparameters"]["mu"], name="FocalLoss")
        else:
            self.L1_loss = L1

        # Set up metrics
        self.L1_metric = tf.keras.metrics.Mean(name="L1")
        self.vq_metric = tf.keras.metrics.Mean(name="vq")
        self.total_metric = tf.keras.metrics.Mean(name="total")

    @property
    def metrics(self):
        return [
            self.L1_metric,
            self.vq_metric,
            self.total_metric
        ]

    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])

        if self.input_times:
            pred, vq = self.UNet.call(source, tf.zeros(1))
        else:
            pred, vq = self.UNet.call(source)

        print("===========================================================")
        print("UNet")
        print("===========================================================")

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

    @tf.function
    def train_step(self, source, target, seg=None, times=None):

        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        # Augmentation if required
        if self.Aug:
            (source, target), seg = self.Aug(imgs=[source, target], seg=seg)

        # Randomise segments of image to sample, get patch indices for each scale
        x, y = self._get_scale_indices()
        source, target, seg = self._sample_patches(x, y, source, target, seg)

        with tf.GradientTape(persistent=True) as tape:

            pred, _ = self(source, times)

            # Calculate L1
            if seg is not None:
                L1_loss = self.L1_loss(target, pred, seg)
            else:
                L1_loss = self.L1_loss(target, pred)

            # Calculate VQ loss
            if self.use_vq:
                vq_loss = sum(self.UNet.losses)
            else:
                vq_loss = 0

            total_loss = L1_loss + vq_loss
            self.L1_metric.update_state(L1_loss)
            self.vq_metric.update_state(vq_loss)
            self.total_metric.update_state(total_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.UNet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.UNet.trainable_variables))

    @tf.function
    def test_step(self, source, target, seg=None, times=None):

        # Get random image patch and generate predicted target
        x, y = self._get_scale_indices()
        source, target, seg = self._sample_patches(x, y, source, target, seg)
        pred, _ = self(source, times)

        # Calculate L1
        if seg is not None:
            L1_loss = self.L1_loss(target, pred, seg)
        else:
            L1_loss = self.L1_loss(target, pred)

        # Calculate VQ loss
        if self.use_vq:
            vq_loss = sum(self.UNet.losses)
        else:
            vq_loss = 0

        total_loss = L1_loss + vq_loss
        self.L1_metric.update_state(L1_loss)
        self.vq_metric.update_state(vq_loss)
        self.total_metric.update_state(total_loss)

    def _get_scale_indices(self):

        # Want higher probability of training on more central regions
        if np.random.randn() > 0.5:
            x = np.random.randint(0, self.scales[0])
            y = np.random.randint(0, self.scales[0])
        else:
            x = np.random.randint(self.scales[0] / 4, self.scales[0] - self.scales[0] / 4)
            y = np.random.randint(self.scales[0] / 4, self.scales[0] - self.scales[0] / 4)

        return x, y

    def _sample_patches(self, x, y, source, target, seg=None):
        x_img = x * self.img_dims[0]
        y_img = y * self.img_dims[1]
        source = source[:, x_img:(x_img + self.img_dims[0]), y_img:(y_img + self.img_dims[1]), :, :]
        target = target[:, x_img:(x_img + self.img_dims[0]), y_img:(y_img + self.img_dims[1]), :, :]

        if seg is not None:
            seg = seg[:, x_img:(x_img + self.img_dims[0]), y_img:(y_img + self.img_dims[1]), :, :]

        return source, target, seg

    def example_inference(self, source, target, seg=None, times=None):
        ex_x = self.scales[0] // 2
        ex_y = self.scales[0] // 2
        source, target, _ = self._sample_patches(ex_x, ex_y, source, target, seg)
        pred, _ = self(source, times)

        return source, target, pred

    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x, t=None):
        return self.UNet(x, t)
