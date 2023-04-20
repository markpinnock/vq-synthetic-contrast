import copy
import enum
from typing import Any

import numpy as np
import tensorflow as tf

from vq_sce.utils.augmentation.augmentation import StdAug

from .components.layers.vq_layers import VQBlock
from .components.unet import MAX_CHANNELS, UNet

# -------------------------------------------------------------------------


@enum.unique
class Task(str, enum.Enum):
    CONTRAST = "contrast"
    SUPER_RES = "super_res"
    DUAL = "dual"
    JOINT = "joint"


# -------------------------------------------------------------------------


class Model(tf.keras.Model):
    """Wrapper for model."""

    def __init__(self, config: dict[str, Any], name: str = "Model") -> None:
        super().__init__(name=name)
        self._initialiser = tf.keras.initializers.HeNormal()
        self._config = config

        self._source_dims = config["data"]["source_dims"]
        self._target_dims = config["data"]["target_dims"]
        config["augmentation"]["source_dims"] = self._source_dims
        config["augmentation"]["target_dims"] = self._target_dims

        self._scales = config["hyperparameters"]["scales"]
        assert len(self._scales) == 1, self._scales
        self._source_dims = [
            self._source_dims[0],
            self._source_dims[1] // self._scales[0],
            self._source_dims[2] // self._scales[0],
        ]
        self._target_dims = [
            self._target_dims[0],
            self._target_dims[1] // self._scales[0],
            self._target_dims[2] // self._scales[0],
        ]
        config["hyperparameters"]["source_dims"] = self._source_dims
        config["hyperparameters"]["target_dims"] = self._target_dims

        if config["hyperparameters"]["vq_layers"] is None:
            self._use_vq = False
        else:
            self._use_vq = True

        # Set up augmentation
        if config["augmentation"]["use"]:
            self.Aug = StdAug(config=config["augmentation"])
        else:
            self.Aug = None

        self.UNet = UNet(self._initialiser, config["hyperparameters"], name="unet")

    def compile(self, optimiser: tf.keras.optimizers.Optimizer) -> None:  # noqa: A003
        super().compile()

        # Set up optimiser and loss
        self.optimiser = optimiser
        self.loss = tf.keras.losses.MeanAbsoluteError()

        # Set up metrics
        prefix = "ce" if self._config["data"]["type"] == "contrast" else "sr"
        self.loss_metric = tf.keras.metrics.Mean(name=f"{prefix}_L1")
        self.vq_metric = tf.keras.metrics.Mean(name=f"{prefix}_vq")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_metric, self.vq_metric]

    def build_model(self) -> None:
        _, _ = self(tf.keras.Input(shape=self._source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._source_dims + [1])
        pred, vq = self.UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

    def train_step(
        self,
        data: dict[str, dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        source = data["source"]
        target = data["target"]

        # Augmentation if required
        if self.Aug:
            (source,), (target,) = self.Aug(source=[source], target=[target])

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape(persistent=True) as tape:
            pred, _ = self(source)

            # Calculate L1
            loss = self.loss(target, pred)

            # Calculate VQ loss
            if self._use_vq:
                vq_loss = sum(self.UNet.losses)
            else:
                vq_loss = 0

            total_loss = loss + vq_loss
            self.loss_metric.update_state(loss)
            self.vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.UNet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.UNet.trainable_variables))

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(
        self,
        data: dict[str, dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        source = data["source"]
        target = data["target"]

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        pred, _ = self(source)

        # Calculate L1
        loss = self.loss(target, pred)

        # Calculate VQ loss
        if self._use_vq:
            vq_loss = sum(self.UNet.losses)
        else:
            vq_loss = 0

        self.loss_metric.update_state(loss)
        self.vq_metric.update_state(vq_loss)

        return {metric.name: metric.result() for metric in self.metrics}

    def _get_scale_indices(self) -> tuple[int, int]:
        # Want higher probability of training on more central regions
        if np.random.randn() > 0.5:
            x = np.random.randint(0, self._scales[0])
            y = np.random.randint(0, self._scales[0])
        else:
            x = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4,
            )
            y = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4,
            )

        return x, y

    def _sample_patches(
        self,
        x: int,
        y: int,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x_src = x * self._source_dims[1]
        y_src = y * self._source_dims[2]
        x_tar = x * self._target_dims[1]
        y_tar = y * self._target_dims[2]
        source = source[
            :,
            :,
            x_src : (x_src + self._source_dims[1]),
            y_src : (y_src + self._source_dims[2]),
            :,
        ]
        target = target[
            :,
            :,
            x_tar : (x_tar + self._target_dims[1]),
            y_tar : (y_tar + self._target_dims[2]),
            :,
        ]

        return source, target

    def example_inference(
        self,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        if self._scales[0] == 1:
            pred, _ = self(source)

        else:
            source, target = self._sample_patches(2, 2, source, target)
            pred, _ = self(source)

        return source, target, pred

    def reset_train_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.UNet(x)


# -------------------------------------------------------------------------


class JointModel(tf.keras.Model):
    """Wrapper for joint super-res/contrast enhancement model."""

    def __init__(self, config: dict[str, Any], name: str = "Model") -> None:
        super().__init__(name=name)
        self._initialiser = tf.keras.initializers.HeNormal()
        self._sr_config = copy.deepcopy(config)
        self._ce_config = copy.deepcopy(config)

        self._sr_source_dims = config["data"]["source_dims"]
        self._sr_target_dims = config["data"]["target_dims"]
        self._ce_source_dims = config["data"]["target_dims"]
        self._ce_target_dims = config["data"]["target_dims"]
        self._sr_config["augmentation"]["source_dims"] = self._sr_source_dims
        self._sr_config["augmentation"]["target_dims"] = self._sr_target_dims
        self._ce_config["augmentation"]["source_dims"] = self._ce_source_dims
        self._ce_config["augmentation"]["target_dims"] = self._ce_target_dims

        self._scales = config["hyperparameters"]["scales"]
        assert len(self._scales) == 1, self._scales
        self._sr_source_dims = [
            self._sr_source_dims[0],
            self._sr_source_dims[1] // self._scales[0],
            self._sr_source_dims[2] // self._scales[0],
        ]
        self._sr_target_dims = [
            self._sr_target_dims[0],
            self._sr_target_dims[1] // self._scales[0],
            self._sr_target_dims[2] // self._scales[0],
        ]
        self._ce_source_dims = [
            self._ce_source_dims[0],
            self._ce_source_dims[1] // self._scales[0],
            self._ce_source_dims[2] // self._scales[0],
        ]
        self._ce_target_dims = [
            self._ce_target_dims[0],
            self._ce_target_dims[1] // self._scales[0],
            self._ce_target_dims[2] // self._scales[0],
        ]

        self._sr_config["hyperparameters"]["source_dims"] = self._sr_source_dims
        self._sr_config["hyperparameters"]["target_dims"] = self._sr_target_dims
        self._ce_config["hyperparameters"]["source_dims"] = self._ce_source_dims
        self._ce_config["hyperparameters"]["target_dims"] = self._ce_target_dims

        assert config["hyperparameters"]["vq_layers"]["bottom"] is not None, config[
            "hyperparameters"
        ]["vq_layers"]
        self._use_vq = True

        # Set up augmentation
        if config["augmentation"]["use"]:
            self.sr_Aug = StdAug(config=self._sr_config["augmentation"])
        else:
            self.sr_Aug = None

        if config["augmentation"]["use"]:
            self.ce_Aug = StdAug(config=self._ce_config["augmentation"])
        else:
            self.ce_Aug = None

        # Get shared VQ layer
        embeddings = config["hyperparameters"]["vq_layers"]["bottom"]
        shared_vq = VQBlock(
            num_embeddings=embeddings,
            embedding_dim=MAX_CHANNELS,
            beta=config["hyperparameters"]["vq_beta"],
            name="shared_vq",
        )

        self.sr_UNet = UNet(
            self._initialiser,
            self._sr_config["hyperparameters"],
            shared_vq=shared_vq,
            name="sr_unet",
        )

        self.ce_UNet = UNet(
            self._initialiser,
            self._ce_config["hyperparameters"],
            shared_vq=shared_vq,
            name="ce_unet",
        )

    def compile(self, optimiser: tf.keras.optimizers.Optimizer) -> None:  # noqa: A003
        super().compile()

        # Set up optimiser and loss
        self.optimiser = optimiser
        self.loss = tf.keras.losses.MeanAbsoluteError()

        # Set up metrics
        self.sr_loss_metric = tf.keras.metrics.Mean(name="sr_L1")
        self.sr_vq_metric = tf.keras.metrics.Mean(name="sr_vq")
        self.ce_loss_metric = tf.keras.metrics.Mean(name="ce_L1")
        self.ce_vq_metric = tf.keras.metrics.Mean(name="ce_vq")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.sr_loss_metric,
            self.sr_vq_metric,
            self.ce_loss_metric,
            self.ce_vq_metric,
        ]

    def build_model(self) -> None:
        _, _ = self(tf.keras.Input(shape=self._sr_source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._sr_source_dims + [1])
        pred, vq = self.sr_UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

        source = tf.keras.Input(shape=self._ce_source_dims + [1])
        pred, vq = self.ce_UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

    def sr_train_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Augmentation if required
        if self.sr_Aug:
            (source,), (target,) = self.sr_Aug(source=[source], target=[target])

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape(persistent=True) as tape:
            pred, _ = self.sr_UNet(source)

            # Calculate L1
            loss = self.loss(target, pred)

            # Calculate VQ loss
            if self._use_vq:
                vq_loss = sum(self.sr_UNet.losses)
            else:
                vq_loss = 0

            total_loss = loss + vq_loss
            self.sr_loss_metric.update_state(loss)
            self.sr_vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.sr_UNet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.sr_UNet.trainable_variables))

    def ce_train_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Augmentation if required
        if self.ce_Aug:
            (source,), (target,) = self.ce_Aug(source=[source], target=[target])

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape(persistent=True) as tape:
            pred, _ = self.ce_UNet(source)

            # Calculate L1
            loss = self.loss(target, pred)

            # Calculate VQ loss
            if self._use_vq:
                vq_loss = sum(self.ce_UNet.losses)
            else:
                vq_loss = 0

            total_loss = loss + vq_loss
            self.ce_loss_metric.update_state(loss)
            self.ce_vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.ce_UNet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.ce_UNet.trainable_variables))

    def train_step(
        self,
        data: dict[str, dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        self.sr_train_step(**data[Task.SUPER_RES])
        self.ce_train_step(**data[Task.CONTRAST])

        return {metric.name: metric.result() for metric in self.metrics}

    def sr_test_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        pred, _ = self.sr_UNet(source)

        # Calculate L1
        loss = self.loss(target, pred)

        # Calculate VQ loss
        if self._use_vq:
            vq_loss = sum(self.sr_UNet.losses)
        else:
            vq_loss = 0

        self.sr_loss_metric.update_state(loss)
        self.sr_vq_metric.update_state(vq_loss)

    def ce_test_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        pred, _ = self.ce_UNet(source)

        # Calculate L1
        loss = self.loss(target, pred)

        # Calculate VQ loss
        if self._use_vq:
            vq_loss = sum(self.ce_UNet.losses)
        else:
            vq_loss = 0

        self.ce_loss_metric.update_state(loss)
        self.ce_vq_metric.update_state(vq_loss)

    def test_step(
        self,
        data: dict[str, dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        self.sr_test_step(**data[Task.SUPER_RES])
        self.ce_test_step(**data[Task.CONTRAST])

        return {metric.name: metric.result() for metric in self.metrics}

    def _get_scale_indices(self) -> tuple[int, int]:
        # Want higher probability of training on more central regions
        if np.random.randn() > 0.5:
            x = np.random.randint(0, self._scales[0])
            y = np.random.randint(0, self._scales[0])
        else:
            x = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4,
            )
            y = np.random.randint(
                self._scales[0] / 4,
                self._scales[0] - self._scales[0] / 4,
            )

        return x, y

    def _sample_patches(
        self,
        x: int,
        y: int,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x_src = x * self._sr_source_dims[1]
        y_src = y * self._sr_source_dims[2]
        x_tar = x * self._sr_target_dims[1]
        y_tar = y * self._sr_target_dims[2]
        source = source[
            :,
            :,
            x_src : (x_src + self._sr_source_dims[1]),
            y_src : (y_src + self._sr_source_dims[2]),
            :,
        ]
        target = target[
            :,
            :,
            x_tar : (x_tar + self._sr_target_dims[1]),
            y_tar : (y_tar + self._sr_target_dims[2]),
            :,
        ]

        return source, target

    def example_inference(
        self,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        if self._scales[0] == 1:
            pred, _ = self(source)

        else:
            source, target = self._sample_patches(2, 2, source, target)
            pred, _ = self(source)

        return source, target, pred

    def reset_train_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x: tf.Tensor, task: str = Task.JOINT) -> tuple[tf.Tensor, None]:
        if task == Task.CONTRAST:
            x, _ = self.ce_UNet(x)
            return x, None

        elif task == Task.SUPER_RES:
            x, _ = self.sr_UNet(x)
            return x, None

        else:
            x, _ = self.sr_UNet(x)
            x, _ = self.ce_UNet(x)
            return x, None


# -------------------------------------------------------------------------


class DualModel(tf.keras.Model):
    """Wrapper for dual super-res/contrast enhancement model (inference only)."""

    def __init__(
        self,
        sr_config: dict[str, Any],
        ce_config: dict[str, Any],
        name: str = "Model",
    ) -> None:
        super().__init__(name=name)
        self._initialiser = tf.keras.initializers.HeNormal()
        self._sr_config = sr_config
        self._ce_config = ce_config

        self._sr_source_dims = sr_config["data"]["source_dims"]
        self._sr_target_dims = sr_config["data"]["target_dims"]
        self._ce_source_dims = ce_config["data"]["target_dims"]
        self._ce_target_dims = ce_config["data"]["target_dims"]

        self._scales = sr_config["hyperparameters"]["scales"]
        assert self._scales == ce_config["hyperparameters"]["scales"]
        assert len(self._scales) == 1, self._scales

        self._sr_source_dims = [
            self._sr_source_dims[0],
            self._sr_source_dims[1] // self._scales[0],
            self._sr_source_dims[2] // self._scales[0],
        ]
        self._sr_target_dims = [
            self._sr_target_dims[0],
            self._sr_target_dims[1] // self._scales[0],
            self._sr_target_dims[2] // self._scales[0],
        ]
        self._ce_source_dims = [
            self._ce_source_dims[0],
            self._ce_source_dims[1] // self._scales[0],
            self._ce_source_dims[2] // self._scales[0],
        ]
        self._ce_target_dims = [
            self._ce_target_dims[0],
            self._ce_target_dims[1] // self._scales[0],
            self._ce_target_dims[2] // self._scales[0],
        ]

        self._sr_config["hyperparameters"]["source_dims"] = self._sr_source_dims
        self._sr_config["hyperparameters"]["target_dims"] = self._sr_target_dims
        self._ce_config["hyperparameters"]["source_dims"] = self._ce_source_dims
        self._ce_config["hyperparameters"]["target_dims"] = self._ce_target_dims

        self.sr_UNet = UNet(
            self._initialiser,
            self._sr_config["hyperparameters"],
            name="sr_unet",
        )

        self.ce_UNet = UNet(
            self._initialiser,
            self._ce_config["hyperparameters"],
            name="ce_unet",
        )

    def build_model(self) -> None:
        _, _ = self(tf.keras.Input(shape=self._sr_source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._sr_source_dims + [1])
        pred, vq = self.sr_UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

        source = tf.keras.Input(shape=self._ce_source_dims + [1])
        pred, vq = self.ce_UNet.call(source)

        if vq is None:
            tf.keras.Model(inputs=source, outputs=pred).summary()
        else:
            tf.keras.Model(inputs=source, outputs=[pred, vq]).summary()

    def call(self, x: tf.Tensor, task: str = Task.DUAL) -> tuple[tf.Tensor, None]:
        if task == Task.CONTRAST:
            x, _ = self.ce_UNet(x)
            return x, None

        elif task == Task.SUPER_RES:
            x, _ = self.sr_UNet(x)
            return x, None

        else:
            x, _ = self.sr_UNet(x)
            x, _ = self.ce_UNet(x)
            return x, None
