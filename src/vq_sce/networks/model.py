import numpy as np
import tensorflow as tf

from .components.unet import UNet
from vq_sce.utils.augmentation.augmentation import StdAug
from vq_sce.utils.losses import L1


#-------------------------------------------------------------------------
""" Wrapper for model """

class Model(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self._initialiser = tf.keras.initializers.HeNormal()
        self._config = config
        self._mb_size = config["expt"]["mb_size"]
        self._source_dims = config["data"]["source_dims"]
        self._target_dims = config["data"]["target_dims"]
        config["hyperparameters"]["source_dims"] = self._source_dims
        config["hyperparameters"]["target_dims"] = self._target_dims
        config["augmentation"]["source_dims"] = self._source_dims
        config["augmentation"]["target_dims"] = self._target_dims
        config["hyperparameters"]["multiscale"] = False

        if config["hyperparameters"]["vq_layers"] is not None:
            self._intermediate_vq = "output" in \
                config["hyperparameters"]["vq_layers"]
        else:
            self._intermediate_vq = False

        self._scales = None # TODO: implement

        if config["hyperparameters"]["vq_layers"] is None:
            self._use_vq = False
        else:
            self._use_vq = True

        # Set up augmentation
        aug_config = config["augmentation"]
        if config["augmentation"]["use"]:
            self.Aug = StdAug(config=aug_config)
        else:
            self.Aug = None

        self.UNet = UNet(
            self._initialiser,
            config["hyperparameters"],
            name="unet"
        )

    def compile(self, optimiser):
        self.optimiser = optimiser

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
        source = tf.keras.Input(shape=self._source_dims + [1])
        pred, vq = self.UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

    @tf.function
    def train_step(self, source, target):

        """ Expects data in order 'source, target'
            or 'source, target, segmentations'
        """

        # Augmentation if required
        if self.Aug:
            (source,), (target,) = self.Aug(source=[source], target=[target])

        with tf.GradientTape(persistent=True) as tape:
            pred, _ = self(source)

            # Calculate L1
            L1_loss = L1(target, pred)

            # Calculate VQ loss
            if self._use_vq:
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
    def test_step(self, source, target):
        pred, _ = self(source)

        # Calculate L1
        L1_loss = L1(target, pred)

        # Calculate VQ loss
        if self._use_vq:
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
            x = np.random.randint(0, self._scales[0])
            y = np.random.randint(0, self._scales[0])
        else:
            x = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4
            )
            y = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4
            )

        return x, y

    def _sample_patches(self, x, y, source, target, seg=None):
        x_src = x * self._source_dims[1]
        y_src = y * self._source_dims[2]
        x_tar = x * self._target_dims[1]
        y_tar = y * self._target_dims[2]
        source = source[
            :,
            :,
            x_src:(x_src + self._source_dims[1]),
            y_src:(y_src + self._source_dims[2]),
            :
        ]
        target = target[
            :,
            :,
            x_tar:(x_tar + self._target_dims[1]),
            y_tar:(y_tar + self._target_dims[2]),
            :
        ]

        if seg is not None:
            seg = seg[
                :,
                :,
                x_tar:(x_tar + self._target_dims[1]),
                y_tar:(y_tar + self._target_dims[2]),
                :
            ]

        return source, target, seg

    def example_inference(self, source, target):
        pred, _ = self(source)

        return source, target, pred

    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x):
        return self.UNet(x)
