import enum
from typing import Any

import tensorflow as tf

from vq_sce.networks.model import JointModel, Task

# -------------------------------------------------------------------------


@enum.unique
class Alpha(int, enum.Enum):
    SUPER_RES = 0
    CONTRAST = 1


class Architect(tf.keras.layers.Layer):
    def __init__(self, name: str = "architect"):
        super().__init__(name=name)

        self.alphas = self.add_weight(
            name=f"{name}/alphas",
            shape=[1, 2],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, alpha_index: int) -> tf.Tensor:
        return tf.keras.activations.softmax(self.alphas)[0][alpha_index]


# -------------------------------------------------------------------------


class DARTSJointModel(JointModel):
    """Wrapper for joint super-res/contrast enhancement model."""

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "DARTSModel",
        dev: bool = False,
    ) -> None:
        super().__init__(config=config, name=name)

        self.architect = Architect(name=f"{name}/architect")

    def compile(  # type: ignore # noqa: A003
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
        self.ce_optimiser = tf.keras.optimizers.Adam(**w_opt_config, name="ce_opt")
        self.alpha_optimiser = tf.keras.optimizers.Adam(
            **a_opt_config, name="alpha_opt"
        )
        self.loss = tf.keras.losses.MeanAbsoluteError()

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
            vq_loss = sum(model.losses)
            loss = self.loss(target, pred) + vq_loss
            loss *= self.architect(alpha_index)

        grads = tape.gradient(loss, model.trainable_variables)
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
        with tf.GradientTape(persistent=True) as tape:
            pred, _ = virtual_model(source)
            vq_loss = sum(virtual_model.losses)
            loss = self.loss(target, pred) + vq_loss
            loss *= self.architect(alpha_index)

        grads = tape.gradient(
            loss,
            virtual_model.trainable_variables + self.architect.trainable_variables,
        )
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
            vq_loss = sum(model.losses)
            loss = self.loss(target, pred) + vq_loss
            loss *= self.architect(alpha_index)
        d_alpha_pos = tape.gradient(loss, self.architect.trainable_variables)
        d_alpha_pos[0] *= mask  # Set non-active alpha gradient to 0

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight.assign_add(-2.0 * epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            vq_loss = sum(model.losses)
            loss = self.loss(target, pred) + vq_loss
            loss *= self.architect(alpha_index)
        d_alpha_neg = tape.gradient(loss, self.architect.trainable_variables)
        d_alpha_neg[0] *= mask  # Set non-active alpha gradient to 0

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight.assign_add(epsilon * grad)

        hessian = [
            (pos - neg) / (2.0 * epsilon) for pos, neg in zip(d_alpha_pos, d_alpha_neg)
        ]

        return hessian

    def architecture_step(
        self,
        virtual_model: tf.keras.Model,
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
            optimiser = self.sr_optimiser
            alpha_index = Alpha.SUPER_RES

        elif task == Task.CONTRAST:
            model = self.ce_UNet
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
