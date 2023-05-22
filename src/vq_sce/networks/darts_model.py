import copy
from typing import Any

import tensorflow as tf

from .model import JointModel, Task

# -------------------------------------------------------------------------


class Architect(tf.keras.layers.Layer):
    def __init__(self, name: str = "Architect"):
        super().__init__(name=name)

        self.alphas = self.add_weight(
            "alphas",
            shape=[2],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self) -> tf.Tensor:
        return tf.keras.activations.softmax(self.alphas)


# -------------------------------------------------------------------------


class DARTSJointModel(JointModel):
    """Wrapper for joint super-res/contrast enhancement model."""

    def __init__(self, config: dict[str, Any], name: str = "DARTSModel") -> None:
        super().__init__(config=config, name=name)

        # Create virtual models to update during architecture search
        self.virtual_sr_UNet = copy.deepcopy(self.sr_UNet)
        self.virtual_ce_UNet = copy.deepcopy(self.ce_UNet)
        self.architect = Architect()
        self.num_alpha = len(self.architect.trainable_variables)

    def compile(  # type: ignore # noqa: A003
        self,
        w_opt_config: dict[str, float],
        a_opt_config: dict[str, float],
    ) -> None:
        super(JointModel, self).compile()
        self.weights_lr = w_opt_config["learning_rate"]

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

    def build_model(self) -> None:
        super().build_model()
        _ = self.sr_UNet(tf.keras.Input(shape=self._sr_source_dims + [1]))
        _ = self.ce_UNet(tf.keras.Input(shape=self._ce_source_dims + [1]))

    def virtual_step(
        self,
        model: tf.keras.Model,
        virtual_model: tf.keras.Model,
        optimiser: tf.keras.optimizers.Optimizer,
        alpha: tf.Tensor,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> None:
        """Get weights that model would have after one training step.
        :param model: model requiring training step
        :param virtual_model: virtual model in which to store updates for above model
        :param optimiser: optimiser for updating virtual model
        :param source: validation source images
        :param target: validation target images
        """
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape() as tape:
            pred, _ = model(source)
            loss = self.loss(target, pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimiser.learning_rate.assign(alpha)
        optimiser.apply_gradients(zip(grads, virtual_model.trainable_variables))
        optimiser.learing_rate.assign(self.weights_lr)

    def unrolled_backward(
        self,
        virtual_model: tf.keras.Model,
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """Calculate gradients for unrolled model on validation data.
        :param virtual_model: virtual model for which to calculate gradients
        :param source: validation source images
        :param target: validation target images
        """
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        with tf.GradientTape() as tape:
            pred, _ = virtual_model(source)
            loss = self.loss(target, pred)

        grads = tape.gradient(
            loss,
            virtual_model.trainable_variables + self.architect.trainable_variables,
        )
        d_weights = grads[self.num_alpha :]
        d_alpha = grads[: self.num_alpha]

        return d_weights, d_alpha

    def calculate_hessian(
        self,
        model: tf.keras.Model,
        d_weights: list[tf.Tensor],
        source: tf.Tensor,
        target: tf.Tensor,
    ) -> list[tf.Tensor]:
        """Calculate Hessian matrix approximation on training data.
        :param model: model to use to calculate loss
        :param model: gradients of model weights
        :param source: training source images
        :param target: training target images
        """
        # Sample patch if needed
        if self._scales[0] > 1:
            x, y = self._get_scale_indices()
            source, target = self._sample_patches(x, y, source, target)

        grad_norm = tf.norm(d_weights)  # stack?
        epsilon = 0.01 / grad_norm

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight += epsilon * grad
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            loss = self.loss(target, pred)
        d_alpha_pos = tape.gradient(loss, self.architect.trainable_variables)

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight -= 2.0 * epsilon * grad
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            loss = self.loss(target, pred)
        d_alpha_neg = tape.gradient(loss, self.architect.trainable_variables)

        for weight, grad in zip(model.trainable_variables, d_weights):
            weight += epsilon * grad

        hessian = [
            (pos - neg) / (2.0 * epsilon) for pos, neg in zip(d_alpha_pos, d_alpha_neg)
        ]

        return hessian

    def architecture_step(
        self,
        data: dict[str, dict[str, tf.Tensor]],
        task: str = Task.JOINT,
    ) -> None:
        if task == Task.SUPER_RES or task == Task.JOINT:
            self.virtual_step(
                model=self.sr_UNet,
                virtual_model=self.virtual_sr_UNet,
                optimiser=self.sr_optimiser,
                **data[Task.SUPER_RES],
            )
            d_weights, d_alpha = self.unrolled_backward(
                virtual_model=self.virtual_sr_UNet,
            )  # val data
            hessian = self.calculate_hessian(
                model=self.sr_UNet, d_weights=d_weights, **data[Task.SUPER_RES]
            )
            arch_grads = []

            for d, h in zip(d_alpha, hessian):
                arch_grads.append(d - self.weights_lr * h)
            self.alpha_optimiser.apply_gradients(
                zip(arch_grads, self.architect.trainable_variables),
            )

        if task == Task.CONTRAST or task == Task.JOINT:
            self.virtual_step(
                model=self.ce_UNet,
                virtual_model=self.virtual_ce_UNet,
                optimiser=self.sr_optimiser,
                **data[Task.CONTRAST],
            )
            d_weights, d_alpha = self.unrolled_backward(
                virtual_model=self.virtual_ce_UNet,
            )  # val data
            hessian = self.calculate_hessian(
                model=self.ce_UNet, d_weights=d_weights, **data[Task.CONTRAST]
            )
            arch_grads = []

            for d, h in zip(d_alpha, hessian):
                arch_grads.append(d - self.weights_lr * h)
            self.alpha_optimiser.apply_gradients(
                zip(arch_grads, self.architect.trainable_variables),
            )

        self.alpha_metric.update_state(self.architect()[0])

    def call(self, x: tf.Tensor, task: str = Task.JOINT) -> tf.Tensor:
        if task == Task.CONTRAST:
            x, _ = self.ce_UNet(x)
            return x

        elif task == Task.SUPER_RES:
            x, _ = self.sr_UNet(x)
            return x

        else:
            x, _ = self.sr_UNet(x)
            x, _ = self.ce_UNet(x)
            return x
