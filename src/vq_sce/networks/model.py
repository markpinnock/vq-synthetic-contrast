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

    def __init__(self, config: dict[str, Any], name: str = "single_model") -> None:
        super().__init__(name=name)
        self._config = config
        self._initialiser = tf.keras.initializers.HeNormal()

        self._local_batch_size = config["expt"]["local_mb_size"]
        self._global_batch_size = config["expt"]["mb_size"]

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

        if config["hyperparameters"]["vq_layers"] is not None:
            vq_blocks = self._get_vq_blocks(config)
        else:
            vq_blocks = {}

        self.UNet = UNet(
            self._initialiser,
            config["hyperparameters"],
            vq_blocks=vq_blocks,
            name="unet",
        )

    def _get_vq_blocks(self, config: dict[str, Any]) -> dict[str, VQBlock]:
        nc = config["hyperparameters"]["nc"]
        vq_layers = config["hyperparameters"]["vq_layers"]
        num_layers = config["hyperparameters"]["layers"]
        vq_blocks = {}

        for layer_name, num_embeddings in vq_layers.items():
            if layer_name == "bottom":
                embedding_dim = np.min([nc * 2 ** (num_layers - 1), MAX_CHANNELS])
            else:
                embedding_dim = np.min([nc * 2 ** (int(layer_name[-1])), MAX_CHANNELS])

            vq_blocks[layer_name] = VQBlock(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                task_lr=1.0,
                beta=config["hyperparameters"]["vq_beta"],
                name=f"vq_{layer_name}",
            )

        return vq_blocks

    def compile(  # noqa: A003
        self,
        opt_config: dict[str, float],
        run_eagerly: bool = False,
    ) -> None:
        super().compile(run_eagerly=run_eagerly)

        # Set up optimiser and loss
        self.optimiser = tf.keras.optimizers.Adam(**opt_config, name="opt")
        self.optimiser = tf.keras.mixed_precision.LossScaleOptimizer(self.optimiser)
        self.loss_object = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM,
        )

        # Set up metrics
        prefix = "ce" if self._config["data"]["type"] == "contrast" else "sr"
        self.loss_metric = tf.keras.metrics.Mean(name=f"{prefix}_L1")
        self.vq_metric = tf.keras.metrics.Mean(name=f"{prefix}_vq")

    def calc_distributed_loss(
        self,
        targets: tf.Tensor,
        preds: tf.Tensor,
        model: tf.keras.Model,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate loss using distributed training strategy."""
        # Calculate L1
        loss = self.loss_object(targets, preds) / tf.cast(
            tf.reduce_prod(self._target_dims),
            tf.float32,
        )
        local_loss = loss / self._local_batch_size
        global_loss = loss / self._global_batch_size

        # Calculate VQ loss
        if self._use_vq:
            local_vq_loss = tf.add_n(model.losses)
            global_vq_loss = tf.nn.scale_regularization_loss(local_vq_loss)
        else:
            local_vq_loss = 0
            global_vq_loss = 0

        return global_loss + global_vq_loss, local_loss, local_vq_loss

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_metric, self.vq_metric]

    def build_model(self) -> None:
        _ = self(tf.keras.Input(shape=self._source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._source_dims + [1])
        pred = self.UNet.call(source)
        tf.keras.Model(inputs=source, outputs=pred).summary()

    def train_step(
        self,
        data: dict[str, tf.Tensor],
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

        with tf.GradientTape() as tape:
            pred = self(source)
            total_loss, loss, vq_loss = self.calc_distributed_loss(
                target,
                pred,
                self.UNet,
            )
            total_loss = self.optimiser.get_scaled_loss(total_loss)

        self.loss_metric.update_state(loss)
        self.vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.UNet.trainable_variables)
        grads = self.optimiser.get_unscaled_gradients(grads)
        self.optimiser.apply_gradients(zip(grads, self.UNet.trainable_variables))

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(
        self,
        data: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        source = data["source"]
        target = data["target"]

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        pred = self(source)
        _, loss, vq_loss = self.calc_distributed_loss(target, pred, self.UNet)

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
        pred = []

        for i in range(source.shape[0]):
            if self._scales[0] == 1:
                pred.append(self(source[i, ...][tf.newaxis, :, :, :, :]))

            else:
                source, target = self._sample_patches(2, 2, source, target)
                pred.append(self(source[i, ...][tf.newaxis, :, :, :, :]))

        return source, target, tf.concat(pred, axis=0)

    def reset_train_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.UNet(x)
        return x


# -------------------------------------------------------------------------


class JointModel(tf.keras.Model):
    """Wrapper for joint super-res/contrast enhancement model."""

    def __init__(self, config: dict[str, Any], name: str = "joint_model") -> None:
        super().__init__(name=name)
        self._initialiser = tf.keras.initializers.HeNormal()
        self._sr_config = copy.deepcopy(config)
        self._ce_config = copy.deepcopy(config)

        assert (
            self._sr_config["expt"]["mb_size"] == self._ce_config["expt"]["mb_size"]
        ), (self._sr_config["expt"]["mb_size"], self._ce_config["expt"]["mb_size"])

        self._local_batch_size = self._sr_config["expt"]["local_mb_size"]
        self._global_batch_size = self._sr_config["expt"]["mb_size"]

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
        shared_vq = self._get_vq_blocks(config)

        self.sr_UNet = UNet(
            self._initialiser,
            self._sr_config["hyperparameters"],
            vq_blocks=shared_vq,
            name="sr_unet",
        )

        self.ce_UNet = UNet(
            self._initialiser,
            self._ce_config["hyperparameters"],
            vq_blocks=shared_vq,
            name="ce_unet",
        )

    def _get_vq_blocks(self, config: dict[str, Any]) -> dict[str, VQBlock]:
        nc = config["hyperparameters"]["nc"]
        vq_layers = config["hyperparameters"]["vq_layers"]
        num_layers = config["hyperparameters"]["layers"]
        vq_blocks = {}

        if config["expt"]["optimisation_type"] in ["darts-task", "darts-both"]:
            # Scale VQ learning rate through DARTS, or...
            task_lr = 1.0
        else:
            # ... halve VQ learning rate as training twice each step
            task_lr = 0.5

        for layer_name, num_embeddings in vq_layers.items():
            if layer_name == "bottom":
                embedding_dim = np.min([nc * 2 ** (num_layers - 1), MAX_CHANNELS])
            else:
                embedding_dim = np.min([nc * 2 ** (int(layer_name[-1])), MAX_CHANNELS])

            vq_blocks[layer_name] = VQBlock(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                task_lr=task_lr,
                beta=config["hyperparameters"]["vq_beta"],
                name=f"vq_{layer_name}",
            )

        return vq_blocks

    def compile(  # noqa: A003
        self,
        opt_config: dict[str, float],
        run_eagerly: bool = False,
    ) -> None:
        super().compile(run_eagerly=run_eagerly)

        # Set up optimiser and loss
        self.sr_optimiser = tf.keras.optimizers.Adam(**opt_config, name="sr_opt")
        self.sr_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.sr_optimiser,
        )
        self.ce_optimiser = tf.keras.optimizers.Adam(**opt_config, name="ce_opt")
        self.ce_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.ce_optimiser,
        )
        self.loss_object = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM,
        )

        # Set up metrics
        self.sr_loss_metric = tf.keras.metrics.Mean(name="sr_L1")
        self.sr_vq_metric = tf.keras.metrics.Mean(name="sr_vq")
        self.ce_loss_metric = tf.keras.metrics.Mean(name="ce_L1")
        self.ce_vq_metric = tf.keras.metrics.Mean(name="ce_vq")

    def calc_distributed_loss(
        self,
        targets: tf.Tensor,
        preds: tf.Tensor,
        model: tf.keras.Model,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate loss using distributed training strategy."""
        # Calculate L1
        loss = self.loss_object(targets, preds) / tf.cast(
            tf.reduce_prod(self._ce_target_dims),
            tf.float32,
        )
        local_loss = loss / self._local_batch_size
        global_loss = loss / self._global_batch_size

        # Calculate VQ loss
        if self._use_vq:
            local_vq_loss = tf.add_n(model.losses)
            global_vq_loss = tf.nn.scale_regularization_loss(local_vq_loss)
        else:
            local_vq_loss = 0
            global_vq_loss = 0

        return global_loss + global_vq_loss, local_loss, local_vq_loss

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.sr_loss_metric,
            self.sr_vq_metric,
            self.ce_loss_metric,
            self.ce_vq_metric,
        ]

    def build_model(self) -> None:
        _ = self(tf.keras.Input(shape=self._sr_source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._sr_source_dims + [1])
        pred = self.sr_UNet.call(source)
        tf.keras.Model(inputs=source, outputs=pred).summary()

        source = tf.keras.Input(shape=self._ce_source_dims + [1])
        pred = self.ce_UNet.call(source)
        tf.keras.Model(inputs=source, outputs=pred).summary()

    def sr_train_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Augmentation if required
        if self.sr_Aug:
            (source,), (target,) = self.sr_Aug(source=[source], target=[target])

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape() as tape:
            pred = self.sr_UNet(source)
            total_loss, loss, vq_loss = self.calc_distributed_loss(
                target,
                pred,
                self.sr_UNet,
            )
            total_loss = self.sr_optimiser.get_scaled_loss(total_loss)

        self.sr_loss_metric.update_state(loss)
        self.sr_vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.sr_UNet.trainable_variables)
        grads = self.sr_optimiser.get_unscaled_gradients(grads)
        self.sr_optimiser.apply_gradients(zip(grads, self.sr_UNet.trainable_variables))

    def ce_train_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Augmentation if required
        if self.ce_Aug:
            (source,), (target,) = self.ce_Aug(source=[source], target=[target])

        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape() as tape:
            pred = self.ce_UNet(source)
            total_loss, loss, vq_loss = self.calc_distributed_loss(
                target,
                pred,
                self.ce_UNet,
            )
            total_loss = self.ce_optimiser.get_scaled_loss(total_loss)

        self.ce_loss_metric.update_state(loss)
        self.ce_vq_metric.update_state(vq_loss)

        # Get gradients and update weights
        grads = tape.gradient(total_loss, self.ce_UNet.trainable_variables)
        grads = self.ce_optimiser.get_unscaled_gradients(grads)
        self.ce_optimiser.apply_gradients(zip(grads, self.ce_UNet.trainable_variables))

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

        pred = self.sr_UNet(source)
        _, loss, vq_loss = self.calc_distributed_loss(target, pred, self.sr_UNet)

        self.sr_loss_metric.update_state(loss)
        self.sr_vq_metric.update_state(vq_loss)

    def ce_test_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        pred = self.ce_UNet(source)
        _, loss, vq_loss = self.calc_distributed_loss(target, pred, self.ce_UNet)

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
        pred = []

        for i in range(source.shape[0]):
            if self._scales[0] == 1:
                pred.append(self(source[i, ...][tf.newaxis, :, :, :, :]))

            else:
                source, target = self._sample_patches(2, 2, source, target)
                pred.append(self(source[i, ...][tf.newaxis, :, :, :, :]))

        return source, target, tf.concat(pred, axis=0)

    def reset_train_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x: tf.Tensor, task: str = Task.JOINT) -> tf.Tensor:
        if task == Task.CONTRAST:
            x = self.ce_UNet(x)
            return x

        elif task == Task.SUPER_RES:
            x = self.sr_UNet(x)
            return x

        else:
            x = self.sr_UNet(x)
            x = self.ce_UNet(x)
            return x


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
            vq_blocks={},
        )

        self.ce_UNet = UNet(
            self._initialiser,
            self._ce_config["hyperparameters"],
            name="ce_unet",
            vq_blocks={},
        )

    def build_model(self) -> None:
        _ = self(tf.keras.Input(shape=self._sr_source_dims + [1]))

    def summary(self) -> None:
        source = tf.keras.Input(shape=self._sr_source_dims + [1])
        pred = self.sr_UNet.call(source)
        tf.keras.Model(inputs=source, outputs=pred).summary()

        source = tf.keras.Input(shape=self._ce_source_dims + [1])
        pred = self.ce_UNet.call(source)
        tf.keras.Model(inputs=source, outputs=pred).summary()

    def call(self, x: tf.Tensor, task: str = Task.DUAL) -> tuple[tf.Tensor, None]:
        if task == Task.CONTRAST:
            x = self.ce_UNet(x)
            return x, None

        elif task == Task.SUPER_RES:
            x = self.sr_UNet(x)
            return x, None

        else:
            x = self.sr_UNet(x)
            x = self.ce_UNet(x)
            return x, None
