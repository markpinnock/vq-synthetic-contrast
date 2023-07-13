import enum
from typing import Any

import tensorflow as tf

from vq_sce.networks.components.layers.vq_layers import DARTSVQBlock
from vq_sce.networks.components.unet import MAX_CHANNELS, UNet
from vq_sce.networks.model import JointModel, Model, Task

# -------------------------------------------------------------------------


@enum.unique
class Alpha(int, enum.Enum):
    SUPER_RES = 0
    CONTRAST = 1


# -------------------------------------------------------------------------


class TaskArchitect(tf.keras.layers.Layer):
    def __init__(self, name: str = "task_architect"):
        super().__init__(name=name, dtype="float32")
        self.softmax = tf.keras.layers.Activation("softmax", dtype="float32")

        self.alphas = self.add_weight(
            name=f"{name}/alphas",
            shape=[1, 2],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype="float32",
        )

    def call(self, alpha_index: int) -> tf.Tensor:
        alpha = self.softmax(self.alphas)
        return alpha[0][alpha_index]


# -------------------------------------------------------------------------


class VQArchitect(tf.keras.layers.Layer):
    def __init__(self, num_dictionaries: int = 0, name: str = "vq_architect"):
        super().__init__(name=name, dtype="float32")
        self.softmax = tf.keras.layers.Activation("softmax", dtype="float32")
        self.num_dictionaries = num_dictionaries

        self.gammas = self.add_weight(
            name=f"{name}/gammas",
            shape=(self.num_dictionaries,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype="float32",
        )

    def call(self) -> tf.Tensor:
        gammas = self.softmax(self.gammas)
        return gammas


# -------------------------------------------------------------------------


class DARTSModel(Model):
    """Wrapper for super-res/contrast enhancement model."""

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "darts_model",
    ) -> None:
        super().__init__(config=config, name=name)

        self.vq_architect = VQArchitect(name=f"{name}/vq_architect")

        # Get virtual VQ layer
        virtual_vq = self._get_vq_block(config)

        # Create virtual model to update during architecture search
        self.virtual_UNet = UNet(
            tf.keras.initializers.Ones(),
            config["hyperparameters"],
            vq_block=virtual_vq,
            name="virtual_unet",
        )

    def _get_vq_block(self, config: dict[str, Any]) -> DARTSVQBlock:
        embeddings = config["hyperparameters"]["vq_layers"]["bottom"]
        shared_vq = DARTSVQBlock(
            num_embeddings=embeddings,
            embedding_dim=MAX_CHANNELS,
            alpha=1.0,
            beta=config["hyperparameters"]["vq_beta"],
            name="shared_vq",
        )
        return shared_vq

    def compile(  # type: ignore[override] # noqa: A003
        self,
        model_opt_config: dict[str, float],
        darts_opt_config: dict[str, float],
        run_eagerly: bool = False,
    ) -> None:
        super(Model, self).compile(run_eagerly=run_eagerly)
        self.weights_lr = model_opt_config["learning_rate"]
        self.run_eagerly = run_eagerly

        # Set up optimiser and loss
        self.optimiser = tf.keras.optimizers.Adam(**model_opt_config, name="model_opt")
        self.optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.optimiser,
        )

        self.darts_optimiser = tf.keras.optimizers.Adam(
            **darts_opt_config, name="darts_opt"
        )
        self.darts_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.darts_optimiser,
        )

        self.loss_object = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM,
        )

        # Set up metrics
        prefix = "ce" if self._config["data"]["type"] == "contrast" else "sr"
        self.loss_metric = tf.keras.metrics.Mean(name=f"{prefix}_L1")
        self.vq_metric = tf.keras.metrics.Mean(name=f"{prefix}_vq")

    def build_model(self) -> None:
        super().build_model()
        _ = self.virtual_UNet(tf.keras.Input(shape=self._source_dims + [1]))

    def virtual_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        """Get weights that model would have after one training step.
        :param source: training source images
        :param target: training target images
        """
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(total_loss, self.UNet.trainable_variables)
        grads = self.optimiser.get_unscaled_gradients(grads)
        self.optimiser.apply_gradients(
            zip(grads, self.virtual_UNet.trainable_variables),
        )

    def unrolled_backward(
        self,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """Calculate gradients for unrolled model on validation data.
        :param source: validation source images
        :param target: validation target images
        """
        with tf.GradientTape() as tape:
            pred, _ = self.virtual_UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(
            total_loss,
            self.virtual_UNet.trainable_variables
            + self.vq_architect.trainable_variables,
        )
        grads = self.darts_optimiser.get_unscaled_gradients(grads)
        num_dict = self.vq_architect.num_dictionaries

        d_weights = grads[:-num_dict]
        d_alpha = grads[-num_dict:]

        return d_weights, d_alpha

    def calculate_hessian(
        self,
        d_weights: list[tf.Tensor],
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> list[tf.Tensor]:
        """Calculate Hessian matrix approximation on training data.
        :param d_weights: gradients of model weights
        :param source: training source images
        :param target: training target images
        """
        grad_norm = tf.sqrt(
            tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in d_weights]),
        )
        epsilon = 0.01 / grad_norm

        for weight, grad in zip(self.UNet.trainable_variables, d_weights):
            weight.assign_add(epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_pos = tape.gradient(total_loss, self.vq_architect.trainable_variables)
        d_alpha_pos = self.darts_optimiser.get_unscaled_gradients(d_alpha_pos)

        for weight, grad in zip(self.UNet.trainable_variables, d_weights):
            weight.assign_add(-2.0 * epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_neg = tape.gradient(total_loss, self.vq_architect.trainable_variables)
        d_alpha_neg = self.darts_optimiser.get_unscaled_gradients(d_alpha_neg)

        for weight, grad in zip(self.UNet.trainable_variables, d_weights):
            weight.assign_add(epsilon * grad)

        hessian = [
            (pos - neg) / (2.0 * epsilon) for pos, neg in zip(d_alpha_pos, d_alpha_neg)
        ]

        return hessian

    def architecture_step(
        self,
        train_data: dict[str, tf.Tensor],
        val_data: dict[str, tf.Tensor],
    ) -> None:
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            train_data["source"], train_data["target"] = self._sample_patches(
                x,
                y,
                train_data["source"],
                train_data["target"],
            )
            val_data["source"], val_data["target"] = self._sample_patches(
                x,
                y,
                val_data["source"],
                val_data["target"],
            )

        self.virtual_step(**train_data)
        d_weights, d_alpha = self.unrolled_backward(**val_data)
        hessian = self.calculate_hessian(d_weights=d_weights, **train_data)

        arch_grads = []
        for d, h in zip(d_alpha, hessian):
            arch_grads.append(d - self.weights_lr * h)

        self.darts_optimiser.apply_gradients(
            zip(arch_grads, self.vq_architect.trainable_variables),
        )

    def train_step(  # type: ignore[override]
        self,
        data: tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        train_batch, valid_batch = data
        self.architecture_step(train_batch, valid_batch)

        self.UNet.bottom_layer.vq.vq_gamma.assign(self.vq_architect())
        self.train_step(**train_batch)

        return {metric.name: metric.result() for metric in self.metrics}


# -------------------------------------------------------------------------


class DARTSJointModel(JointModel):
    """Wrapper for joint super-res/contrast enhancement model."""

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "darts_model",
    ) -> None:
        super().__init__(config=config, name=name)

        if config["expt"]["optimisation_type"] in ["darts-task", "darts-both"]:
            self.task_architect = TaskArchitect(name=f"{name}/task_architect")
        else:
            self.task_architect = None

        if config["expt"]["optimisation_type"] in ["darts-vq", "darts-both"]:
            self.vq_architect = VQArchitect(name=f"{name}/vq_architect")
        else:
            self.vq_architect = None

        assert self.task_architect and self.vq_architect

        # Get shared VQ layer
        virtual_shared_vq = self._get_vq_block(config)

        # Create virtual models to update during architecture search
        self.virtual_sr_UNet = UNet(
            tf.keras.initializers.Ones(),
            self._sr_config["hyperparameters"],
            vq_block=virtual_shared_vq,
            name="virtual_sr_unet",
        )

        self.virtual_ce_UNet = UNet(
            tf.keras.initializers.Ones(),
            self._ce_config["hyperparameters"],
            vq_block=virtual_shared_vq,
            name="virtual_ce_unet",
        )

    def compile(  # type: ignore[override] # noqa: A003
        self,
        w_opt_config: dict[str, float],
        a_opt_config: dict[str, float],
        run_eagerly: bool = False,
    ) -> None:
        super(JointModel, self).compile(run_eagerly=run_eagerly)
        self.weights_lr = w_opt_config["learning_rate"]
        self.run_eagerly = run_eagerly

        # Set up optimiser and loss
        self.sr_optimiser = tf.keras.optimizers.Adam(**w_opt_config, name="sr_opt")
        self.sr_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.sr_optimiser,
        )
        self.ce_optimiser = tf.keras.optimizers.Adam(**w_opt_config, name="ce_opt")
        self.ce_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.ce_optimiser,
        )

        self.alpha_optimiser = tf.keras.optimizers.Adam(
            **a_opt_config, name="alpha_opt"
        )
        self.alpha_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.alpha_optimiser,
        )

        self.loss_object = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM,
        )

        # Set up metrics
        self.sr_loss_metric = tf.keras.metrics.Mean(name="sr_L1")
        self.sr_vq_metric = tf.keras.metrics.Mean(name="sr_vq")
        self.ce_loss_metric = tf.keras.metrics.Mean(name="ce_L1")
        self.ce_vq_metric = tf.keras.metrics.Mean(name="ce_vq")
        self.alpha_metric = tf.keras.metrics.Mean(name="alpha")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.sr_loss_metric,
            self.sr_vq_metric,
            self.ce_loss_metric,
            self.ce_vq_metric,
            self.alpha_metric,
        ]

    def build_model(self) -> None:
        super().build_model()
        _ = self.virtual_sr_UNet(tf.keras.Input(shape=self._sr_source_dims + [1]))
        _ = self.virtual_ce_UNet(tf.keras.Input(shape=self._ce_source_dims + [1]))

    def virtual_step(
        self,
        model: tf.keras.Model,
        virtual_model: tf.keras.Model,
        optimiser: tf.keras.optimizers.Optimizer,
        alpha_index: int,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> None:
        """Get weights that model would have after one training step.
        :param model: model requiring training step
        :param virtual_model: virtual model in which to store updates for above model
        :param optimiser: optimiser for updating virtual model
        :param alpha_index: index of loss weighting for this task
        :param source: training source images
        :param target: training target images
        """
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                model,
            )
            total_loss *= self.architect(alpha_index)
            total_loss = optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads = optimiser.get_unscaled_gradients(grads)
        optimiser.apply_gradients(zip(grads, virtual_model.trainable_variables))

    def unrolled_backward(
        self,
        virtual_model: tf.keras.Model,
        alpha_index: tf.Tensor,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """Calculate gradients for unrolled model on validation data.
        :param virtual_model: virtual model for which to calculate gradients
        :param alpha_index: index of loss weighting for this task
        :param source: validation source images
        :param target: validation target images
        """
        with tf.GradientTape() as tape:
            pred, _ = virtual_model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                virtual_model,
            )
            total_loss *= self.architect(alpha_index)
            total_loss = self.alpha_optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(
            total_loss,
            virtual_model.trainable_variables + self.architect.trainable_variables,
        )
        grads = self.alpha_optimiser.get_unscaled_gradients(grads)

        d_weights = grads[:-1]
        d_alpha = grads[-1:]
        mask = tf.constant([[1.0 - alpha_index, alpha_index]])
        d_alpha[0] *= mask  # Set non-active alpha gradient to 0

        return d_weights, d_alpha

    def calculate_hessian(
        self,
        model: tf.keras.Model,
        d_weights: list[tf.Tensor],
        alpha_index: tf.Tensor,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> list[tf.Tensor]:
        """Calculate Hessian matrix approximation on training data.
        :param model: model to use to calculate loss
        :param d_weights: gradients of model weights
        :param alpha_index: index of loss weighting for this task
        :param source: training source images
        :param target: training target images
        """
        mask = tf.constant([[1.0 - alpha_index, alpha_index]])

        grad_norm = tf.sqrt(
            tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in d_weights]),
        )
        epsilon = 0.01 / grad_norm

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight.assign_add(epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                model,
            )
            total_loss *= self.architect(alpha_index)
            total_loss = self.alpha_optimiser.get_scaled_loss(total_loss)

        d_alpha_pos = tape.gradient(total_loss, self.architect.trainable_variables)
        d_alpha_pos = self.alpha_optimiser.get_unscaled_gradients(d_alpha_pos)
        d_alpha_pos[0] *= mask  # Set non-active alpha gradient to 0

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight.assign_add(-2.0 * epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                model,
            )
            total_loss *= self.architect(alpha_index)
            total_loss = self.alpha_optimiser.get_scaled_loss(total_loss)

        d_alpha_neg = tape.gradient(total_loss, self.architect.trainable_variables)
        d_alpha_neg = self.alpha_optimiser.get_unscaled_gradients(d_alpha_neg)
        d_alpha_neg[0] *= mask  # Set non-active alpha gradient to 0

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight.assign_add(epsilon * grad)

        hessian = [
            (pos - neg) / (2.0 * epsilon) for pos, neg in zip(d_alpha_pos, d_alpha_neg)
        ]

        return hessian

    def architecture_step(
        self,
        train_data: dict[str, tf.Tensor],
        val_data: dict[str, tf.Tensor],
        task: str,
    ) -> None:
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            train_data["source"], train_data["target"] = self._sample_patches(
                x,
                y,
                train_data["source"],
                train_data["target"],
            )
            val_data["source"], val_data["target"] = self._sample_patches(
                x,
                y,
                val_data["source"],
                val_data["target"],
            )

        if task == Task.SUPER_RES:
            model = self.sr_UNet
            virtual_model = self.virtual_sr_UNet
            optimiser = self.sr_optimiser
            alpha_index = Alpha.SUPER_RES

        elif task == Task.CONTRAST:
            model = self.ce_UNet
            virtual_model = self.virtual_ce_UNet
            optimiser = self.ce_optimiser
            alpha_index = Alpha.SUPER_RES

        else:
            raise ValueError(f"Invalid task: {task}")

        self.virtual_step(
            model=model,
            virtual_model=virtual_model,
            alpha_index=alpha_index,
            optimiser=optimiser,
            **train_data,
        )
        d_weights, d_alpha = self.unrolled_backward(
            virtual_model=virtual_model,
            alpha_index=alpha_index,
            **val_data,
        )
        hessian = self.calculate_hessian(
            model=model,
            d_weights=d_weights,
            alpha_index=alpha_index,
            **train_data,
        )

        arch_grads = []
        for d, h in zip(d_alpha, hessian):
            arch_grads.append(d - self.weights_lr * h)

        self.alpha_optimiser.apply_gradients(
            zip(arch_grads, self.architect.trainable_variables),
        )

    def train_step(  # type: ignore[override]
        self,
        data: tuple[dict[str, dict[str, tf.Tensor]], dict[str, dict[str, tf.Tensor]]],
    ) -> dict[str, tf.Tensor]:
        train_batch, valid_batch = data
        self.architecture_step(
            train_batch[Task.SUPER_RES],
            valid_batch[Task.SUPER_RES],
            Task.SUPER_RES,
        )
        self.architecture_step(
            train_batch[Task.CONTRAST],
            valid_batch[Task.CONTRAST],
            Task.CONTRAST,
        )
        self.alpha_metric.update_state(self.architect(Alpha.SUPER_RES))

        self.shared_vq.vq_alpha.assign(self.architect(Alpha.SUPER_RES))
        self.sr_train_step(**train_batch[Task.SUPER_RES])
        self.shared_vq.vq_alpha.assign(self.architect(Alpha.CONTRAST))
        self.ce_train_step(**train_batch[Task.CONTRAST])

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, x: tf.Tensor, task: str = Task.JOINT) -> tf.Tensor:
        if task == Task.CONTRAST:
            self.shared_vq.vq_alpha.assign(self.architect(Alpha.CONTRAST))
            x, _ = self.ce_UNet(x)
            return x

        elif task == Task.SUPER_RES:
            self.shared_vq.vq_alpha.assign(self.architect(Alpha.SUPER_RES))
            x, _ = self.sr_UNet(x)
            return x

        else:
            self.shared_vq.vq_alpha.assign(self.architect(Alpha.SUPER_RES))
            x, _ = self.sr_UNet(x)
            self.shared_vq.vq_alpha.assign(self.architect(Alpha.CONTRAST))
            x, _ = self.ce_UNet(x)
            return x
