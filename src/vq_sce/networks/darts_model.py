from typing import Any

import tensorflow as tf

from vq_sce.networks.components.layers.vq_layers import DARTSVQBlock, VQBlock
from vq_sce.networks.components.unet import MAX_CHANNELS, UNet
from vq_sce.networks.model import JointModel, Model, Task

# -------------------------------------------------------------------------


class DARTSModel(Model):
    """Wrapper for super-res/contrast enhancement model."""

    unet_variables: list[tf.Variable]
    virtual_unet_variables: list[tf.Variable]

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "darts_model",
    ) -> None:
        super().__init__(config=config, name=name)
        self.num_dict = self.UNet.vq_block.num_dictionaries

        # Weights for candidate VQ dictionaries
        self.alpha_vq = tf.Variable(
            tf.zeros((1, self.num_dict)),
            name=f"{name}/alpha_vq",
        )
        self.UNet.vq_block.alpha_vq = self.alpha_vq

        # Get virtual VQ layer
        virtual_vq = self._get_vq_block(config)

        # Create virtual model to update during architecture search
        self.virtual_UNet = UNet(
            tf.keras.initializers.Ones(),
            config["hyperparameters"],
            vq_block=virtual_vq,
            name="virtual_unet",
        )
        self.virtual_UNet.vq_block.alpha_vq = self.alpha_vq

    def _get_vq_block(self, config: dict[str, Any]) -> DARTSVQBlock:
        embeddings = config["hyperparameters"]["vq_layers"]["bottom"]
        vq = DARTSVQBlock(
            num_embeddings=embeddings,
            embedding_dim=MAX_CHANNELS,
            task_lr=1.0,
            beta=config["hyperparameters"]["vq_beta"],
            name="virtual_vq",
        )
        return vq

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
        """Initialise variables in models.
        This sub-class also ensures that the DARTS parameters
        are not included in the model trainable variables.
        """
        super().build_model()
        self.unet_variables = [
            v for v in self.UNet.trainable_variables if "alpha_vq" not in v.name
        ]

        _ = self.virtual_UNet(tf.keras.Input(shape=self._source_dims + [1]))
        self.virtual_unet_variables = [
            v for v in self.virtual_UNet.trainable_variables if "alpha_vq" not in v.name
        ]

    def virtual_step(self, source: tf.Tensor, target: tf.Tensor) -> None:
        """Get weights that model would have after one training step.
        :param source: training source images
        :param target: training target images
        """
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(total_loss, self.unet_variables)
        grads = self.optimiser.get_unscaled_gradients(grads)
        self.optimiser.apply_gradients(
            zip(grads, self.virtual_unet_variables),
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
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                self.virtual_UNet,
            )
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(
            total_loss,
            self.virtual_unet_variables + [self.alpha_vq],
        )
        grads = self.darts_optimiser.get_unscaled_gradients(grads)

        d_weights = grads[:-1]
        d_alpha = grads[-1:]

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

        for weight, grad in zip(self.unet_variables, d_weights):
            weight.assign_add(epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_pos = tape.gradient(total_loss, [self.alpha_vq])
        d_alpha_pos = self.darts_optimiser.get_unscaled_gradients(d_alpha_pos)

        for weight, grad in zip(self.unet_variables, d_weights):
            weight.assign_add(-2.0 * epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = self.UNet(source)
            total_loss, _, _ = self.calc_distributed_loss(target, pred, self.UNet)
            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_neg = tape.gradient(total_loss, [self.alpha_vq])
        d_alpha_neg = self.darts_optimiser.get_unscaled_gradients(d_alpha_neg)

        for weight, grad in zip(self.unet_variables, d_weights):
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

        alpha_grads = []
        for d, h in zip(d_alpha, hessian):
            alpha_grads.append(d - self.weights_lr * h)

        self.darts_optimiser.apply_gradients(zip(alpha_grads, [self.alpha_vq]))

    def train_step(  # type: ignore[override]
        self,
        data: tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        train_batch, valid_batch = data

        self.architecture_step(train_batch, valid_batch)
        super().train_step(train_batch)

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
        self.num_alpha_variables = 0

        # Weights for candidate VQ dictionaries
        if isinstance(config["hyperparameters"]["vq_layers"]["bottom"], list):
            self.num_dict = self.sr_UNet.vq_block.num_dictionaries
            self.alpha_vq = [
                tf.Variable(
                    tf.zeros((1, self.num_dict)),
                    name=f"{name}/alpha_vq",
                ),
            ]
            self.sr_UNet.vq_block.alpha_vq = self.alpha_vq[0]
            self.num_alpha_variables += 1

        else:
            self.alpha_vq = []

        # Weights for tasks
        if config["expt"]["optimisation_type"] in ["darts-task", "darts-both"]:
            self.alpha_task = [tf.Variable(0.0, name=f"{name}/alpha_task")]
            self.num_alpha_variables += 1

        else:
            self.alpha_task = []

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

        if isinstance(config["hyperparameters"]["vq_layers"]["bottom"], list):
            self.virtual_sr_UNet.vq_block.alpha_vq = self.alpha_vq[0]

    def _get_vq_block(self, config: dict[str, Any]) -> VQBlock | DARTSVQBlock:
        embeddings = config["hyperparameters"]["vq_layers"]["bottom"]

        if config["expt"]["optimisation_type"] in ["darts-task", "darts-both"]:
            # Scale VQ learning rate through DARTS, or...
            task_lr = 1.0
        else:
            # ... halve VQ learning rate as training twice each step
            task_lr = 0.5

        if isinstance(config["hyperparameters"]["vq_layers"]["bottom"], list):
            vq = DARTSVQBlock(
                num_embeddings=embeddings,
                embedding_dim=MAX_CHANNELS,
                task_lr=task_lr,
                beta=config["hyperparameters"]["vq_beta"],
                name="shared_vq",
            )

        else:
            vq = VQBlock(
                num_embeddings=embeddings,
                embedding_dim=MAX_CHANNELS,
                task_lr=task_lr,
                beta=config["hyperparameters"]["vq_beta"],
                name="shared_vq",
            )

        return vq

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

        self.darts_optimiser = tf.keras.optimizers.Adam(
            **a_opt_config, name="alpha_opt"
        )
        self.darts_optimiser = tf.keras.mixed_precision.LossScaleOptimizer(
            self.darts_optimiser,
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
        """Initialise variables in models.
        This sub-class also ensures that the DARTS parameters
        are not included in the model trainable variables.
        """
        super().build_model()
        self.sr_unet_variables = [
            v for v in self.sr_UNet.trainable_variables if "alpha" not in v.name
        ]
        self.ce_unet_variables = [
            v for v in self.ce_UNet.trainable_variables if "alpha" not in v.name
        ]

        _ = self.virtual_sr_UNet(tf.keras.Input(shape=self._sr_source_dims + [1]))
        _ = self.virtual_ce_UNet(tf.keras.Input(shape=self._ce_source_dims + [1]))
        self.virtual_sr_unet_variables = [
            v for v in self.virtual_sr_UNet.trainable_variables if "alpha" not in v.name
        ]
        self.virtual_ce_unet_variables = [
            v for v in self.virtual_ce_UNet.trainable_variables if "alpha" not in v.name
        ]

    def virtual_step(
        self,
        model: tf.keras.Model,
        model_variables: list[tf.Tensor],
        virtual_model_variables: list[tf.Tensor],
        optimiser: tf.keras.optimizers.Optimizer,
        task: str,
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
            if len(self.alpha_task) == 1:
                if task == Task.SUPER_RES:
                    total_loss *= tf.nn.sigmoid(self.alpha_task[0])
                else:
                    total_loss *= 1 - tf.nn.sigmoid(self.alpha_task[0])

            total_loss = optimiser.get_scaled_loss(total_loss)

        grads = tape.gradient(total_loss, model_variables)
        grads = optimiser.get_unscaled_gradients(grads)
        optimiser.apply_gradients(zip(grads, virtual_model_variables))

    def unrolled_backward(
        self,
        virtual_model: tf.keras.Model,
        virtual_model_variables: list[tf.Tensor],
        task: str,
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
            if len(self.alpha_task) == 1:
                if task == Task.SUPER_RES:
                    total_loss *= tf.nn.sigmoid(self.alpha_task[0])
                else:
                    total_loss *= 1 - tf.nn.sigmoid(self.alpha_task[0])

            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        vq_variables = self.alpha_vq + self.alpha_task

        grads = tape.gradient(
            total_loss,
            virtual_model_variables + vq_variables,
        )
        grads = self.darts_optimiser.get_unscaled_gradients(grads)

        d_weights = grads[: -self.num_alpha_variables]
        d_alpha = grads[-self.num_alpha_variables :]

        return d_weights, d_alpha

    def calculate_hessian(
        self,
        model: tf.keras.Model,
        model_variables: list[tf.Tensor],
        d_weights: list[tf.Tensor],
        task: str,
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
        grad_norm = tf.sqrt(
            tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in d_weights]),
        )
        epsilon = 0.01 / grad_norm
        vq_variables = self.alpha_vq + self.alpha_task

        for weight, grad in zip(model_variables, d_weights):
            weight.assign_add(epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                model,
            )
            if len(self.alpha_task) == 1:
                if task == Task.SUPER_RES:
                    total_loss *= tf.nn.sigmoid(self.alpha_task[0])
                else:
                    total_loss *= 1 - tf.nn.sigmoid(self.alpha_task[0])

            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_pos = tape.gradient(total_loss, vq_variables)
        d_alpha_pos = self.darts_optimiser.get_unscaled_gradients(d_alpha_pos)

        for weight, grad in zip(model_variables, d_weights):
            weight.assign_add(-2.0 * epsilon * grad)
        with tf.GradientTape() as tape:
            pred, _ = model(source)
            total_loss, _, _ = self.calc_distributed_loss(
                target,
                pred,
                model,
            )
            if len(self.alpha_task) == 1:
                if task == Task.SUPER_RES:
                    total_loss *= tf.nn.sigmoid(self.alpha_task[0])
                else:
                    total_loss *= 1 - tf.nn.sigmoid(self.alpha_task[0])

            total_loss = self.darts_optimiser.get_scaled_loss(total_loss)

        d_alpha_neg = tape.gradient(total_loss, vq_variables)
        d_alpha_neg = self.darts_optimiser.get_unscaled_gradients(d_alpha_neg)

        for weight, grad in zip(model_variables, d_weights):
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
            model_variables = self.sr_unet_variables
            virtual_model = self.virtual_sr_UNet
            virtual_model_variables = self.virtual_sr_unet_variables
            optimiser = self.sr_optimiser

        elif task == Task.CONTRAST:
            model = self.ce_UNet
            model_variables = self.ce_unet_variables
            virtual_model = self.virtual_ce_UNet
            virtual_model_variables = self.virtual_ce_unet_variables
            optimiser = self.ce_optimiser

        else:
            raise ValueError(f"Invalid task: {task}")

        self.virtual_step(
            model=model,
            model_variables=model_variables,
            virtual_model_variables=virtual_model_variables,
            optimiser=optimiser,
            task=task,
            **train_data,
        )
        d_weights, d_alpha = self.unrolled_backward(
            virtual_model=virtual_model,
            virtual_model_variables=virtual_model_variables,
            task=task,
            **val_data,
        )
        hessian = self.calculate_hessian(
            model=model,
            model_variables=model_variables,
            d_weights=d_weights,
            task=task,
            **train_data,
        )

        arch_grads = []
        vq_variables = self.alpha_vq + self.alpha_task
        for d, h in zip(d_alpha, hessian):
            arch_grads.append(d - self.weights_lr * h)

        self.darts_optimiser.apply_gradients(zip(arch_grads, vq_variables))

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
        self.alpha_metric.update_state(tf.nn.sigmoid(self.alpha_task))
        self.sr_train_step(**train_batch[Task.SUPER_RES])
        self.ce_train_step(**train_batch[Task.CONTRAST])

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, x: tf.Tensor, task: str = Task.JOINT) -> tf.Tensor:
        if task == Task.SUPER_RES:
            x, _ = self.ce_UNet(x)
            return x

        elif task == Task.CONTRAST:
            x, _ = self.sr_UNet(x)
            return x

        else:
            x, _ = self.sr_UNet(x)
            x, _ = self.ce_UNet(x)
            return x
