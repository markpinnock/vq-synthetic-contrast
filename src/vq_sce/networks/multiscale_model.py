import copy
import numpy as np
import tensorflow as tf

from vq_sce import LQ_SLICE_THICK
from .components.unet import MultiscaleUNet, MAX_CHANNELS
from .components.layers.vq_layers import VQBlock
from vq_sce.utils.augmentation.augmentation import StdAug
from vq_sce.utils.losses import L1


#-------------------------------------------------------------------------
""" Wrapper for multi-scale U-Net """

class MultiscaleModel(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self._config = config

        self._source_dims = config["data"]["source_dims"]
        self._target_dims = config["data"]["target_dims"]
        config["augmentation"]["source_dims"] = self._source_dims
        config["augmentation"]["target_dims"] = self._target_dims

        self._scales = config["hyperparameters"]["scales"]
        self._source_dims = [
            self._source_dims[0],
            self._source_dims[1] // self._scales[0],
            self._source_dims[2] // self._scales[0]
        ]
        self._target_dims = [
            self._target_dims[0],
            self._target_dims[1] // self._scales[0],
            self._target_dims[2] // self._scales[0]
        ]

        config["hyperparameters"]["source_dims"] = self._source_dims
        config["hyperparameters"]["target_dims"] = self._target_dims

        if config["hyperparameters"]["vq_layers"] is not None:
            self.intermediate_vq = "output" in \
                config["hyperparameters"]["vq_layers"]
        else:
            self.intermediate_vq = False

        if config["hyperparameters"]["vq_layers"] is None:
            self.use_vq = False
        else:
            self.use_vq = True

        # Set up augmentation
        if config["augmentation"]["use"]:
            self.Aug = StdAug(config=config["augmentation"])
        else:
            self.Aug = None

        self.UNet = MultiscaleUNet(
            self.initialiser,
            config["hyperparameters"],
            name="unet"
        )

    def compile(self, optimiser):
        self.optimiser = optimiser

        # Set up metrics
        prefix = "ce" if self._config["data"]["type"] == "contrast" else "sr"
        self.L1_metric = tf.keras.metrics.Mean(name=f"{prefix}_L1")
        self.vq_metric = tf.keras.metrics.Mean(name=f"{prefix}_vq")
        self.total_metric = tf.keras.metrics.Mean(name=f"{prefix}_total")

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

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self._scales[0]))
        ys = list(range(self._scales[0]))

        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, _ = self._sample_patches(target_x, target_y, target)

                with tf.GradientTape(persistent=True) as tape:
                    # Perform multi-scale training
                    source_patch, _ = self._sample_patches(x[0], y[0], source)
                    pred, vq = self(source_patch)

                    for i in range(1, len(self._scales) - 1):
                        pred, vq = self._sample_patches(x[i], y[i], pred, vq)

                        if self._config["data"]["type"] == "super_res":
                            pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

                        if self.intermediate_vq:
                            pred, vq = self(vq)
                        else:
                            pred, vq = self(pred)

                    pred, _ = self._sample_patches(x[-1], y[-1], pred)
                    
                    # Calculate L1
                    L1_loss = L1(target_patch, pred)

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
    def test_step(self, source, target):

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self._scales[0]))
        ys = list(range(self._scales[0]))

        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, _ = self._sample_patches(target_x, target_y, target)

                # Perform multi-scale inference
                source_patch, _ = self._sample_patches(x[0], y[0], source)
                pred, vq = self(source_patch)

                for i in range(1, len(self._scales) - 1):
                    pred, vq = self._sample_patches(x[i], y[i], pred, vq)

                    if self._config["data"]["type"] == "super_res":
                        pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

                    if self.intermediate_vq:
                        pred, vq = self(vq)
                    else:
                        pred, vq = self(pred)

                pred, _ = self._sample_patches(x[-1], y[-1], pred)

                # Calculate L1
                L1_loss = L1(target_patch, pred)

                # Calculate VQ loss
                if self.use_vq:
                    vq_loss = sum(self.UNet.losses)
                else:
                    vq_loss = 0

                total_loss = L1_loss + vq_loss
                self.L1_metric.update_state(L1_loss)
                self.vq_metric.update_state(vq_loss)
                self.total_metric.update_state(total_loss)

    def _get_scale_indices(self, target_x=None, target_y=None):
        if target_x is None or target_y is None:
            # Want higher probability of training on more central regions
            if np.random.randn() > 0.5:
                target_x = np.random.randint(0, self._scales[0])
                target_y = np.random.randint(0, self._scales[0])
            else:
                target_x = np.random.randint(
                    self._scales[0] / 4,
                    self._scales[0] - self._scales[0] / 4
                )
                target_y = np.random.randint(
                    self._scales[0] / 4,
                    self._scales[0] - self._scales[0] / 4
                )

        binary_rep = bin(target_x)[2:]
        source_x = [0 for _ in range(len(self._scales) - len(binary_rep))]
        for c in binary_rep:
            source_x.append(int(c))

        binary_rep = bin(target_y)[2:]
        source_y = [0 for _ in range(len(self._scales) - len(binary_rep))]
        for c in binary_rep:
            source_y.append(int(c))

        return source_x, source_y, target_x, target_y

    def _downsample_images(self, img):
        img = img[:, :, ::self._scales[0], ::self._scales[0], :]
        return img

    def _down_sample_super_res(self, img):
        img = img[:, 1::LQ_SLICE_THICK, :, :, :]
        return img

    def _sample_patches(self, x, y, img1, img2=None):
        x_img = x * self._source_dims[1]
        y_img = y * self._source_dims[2]
        img1 = img1[:, :, x_img:(x_img + self._source_dims[1]), y_img:(y_img + self._source_dims[2]), :]

        if img2 is not None:
            img2 = img2[:, :, x_img:(x_img + self._source_dims[1]), y_img:(y_img + self._source_dims[2]), :]

        return img1, img2

    def example_inference(self, source, target):
        x, y, target_x, target_y = self._get_scale_indices(
            self._scales[0] // 2,
            self._scales[0] // 2
        )
        preds = {}
        source_patch, target = self._sample_patches(target_x, target_y, source, target)
        source = self._downsample_images(source)
        source, _ = self._sample_patches(x[0], y[0], source)
        pred, vq = self(source)
        preds[str(self._scales[0])] = pred

        for i in range(1, len(self._scales) - 1):
            pred, vq = self._sample_patches(x[i], y[i], pred, vq)

            if self._config["data"]["type"] == "super_res":
                pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

            if self.intermediate_vq:
                pred, vq = self(vq)
                preds[str(self._scales[i])] = vq
            else:
                pred, vq = self(pred)
                preds[str(self._scales[i])] = pred

        pred, vq = self._sample_patches(x[-1], y[-1], pred, vq)
        preds[str(self._scales[-1])] = pred

        return source_patch, target, preds

    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x):
        return self.UNet(x)


#-------------------------------------------------------------------------
""" Wrapper for multi-scale joint super-res/contrast enhancement model """

class JointMultiscaleModel(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
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
            self._sr_source_dims[2] // self._scales[0]
        ]
        self._sr_target_dims = [
            self._sr_target_dims[0],
            self._sr_target_dims[1] // self._scales[0],
            self._sr_target_dims[2] // self._scales[0]
        ]
        self._ce_source_dims = [
            self._ce_source_dims[0],
            self._ce_source_dims[1] // self._scales[0],
            self._ce_source_dims[2] // self._scales[0]
        ]
        self._ce_target_dims = [
            self._ce_target_dims[0],
            self._ce_target_dims[1] // self._scales[0],
            self._ce_target_dims[2] // self._scales[0]
        ]

        self._sr_config["hyperparameters"]["source_dims"] = self._sr_source_dims
        self._sr_config["hyperparameters"]["target_dims"] = self._sr_target_dims
        self._ce_config["hyperparameters"]["source_dims"] = self._ce_source_dims
        self._ce_config["hyperparameters"]["target_dims"] = self._ce_target_dims

        assert config["hyperparameters"]["vq_layers"]["bottom"] is not None, (
            config["hyperparameters"]["vq_layers"]
        )
        self._use_vq = True

        if config["hyperparameters"]["vq_layers"] is not None:
            self.intermediate_vq = "output" in \
                config["hyperparameters"]["vq_layers"]
        else:
            self.intermediate_vq = False

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
            name="shared_vq"
        )

        self.sr_UNet = MultiscaleUNet(
            self._initialiser,
            self._sr_config["hyperparameters"],
            shared_vq=shared_vq,
            name="sr_unet"
        )

        self.ce_UNet = MultiscaleUNet(
            self._initialiser,
            self._ce_config["hyperparameters"],
            shared_vq=shared_vq,
            name="ce_unet"
        )

    def compile(self, optimiser):
        self.optimiser = optimiser

        # Set up metrics
        self.sr_L1_metric = tf.keras.metrics.Mean(name="sr_L1")
        self.sr_vq_metric = tf.keras.metrics.Mean(name="sr_vq")
        self.sr_total_metric = tf.keras.metrics.Mean(name="sr_total")
        self.ce_L1_metric = tf.keras.metrics.Mean(name="ce_L1")
        self.ce_vq_metric = tf.keras.metrics.Mean(name="ce_vq")
        self.ce_total_metric = tf.keras.metrics.Mean(name="ce_total")

    @property
    def metrics(self):
        return [
            self.sr_L1_metric,
            self.sr_vq_metric,
            self.sr_total_metric,
            self.ce_L1_metric,
            self.ce_vq_metric,
            self.ce_total_metric
        ]

    def summary(self):
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

    @tf.function
    def train_step(self, source, target):

        """ Expects data in order 'source, target'
            or 'source, target, segmentations'
        """

        # Augmentation if required
        if self.Aug:
            (source,), (target,) = self.Aug(source=[source], target=[target])

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self._scales[0]))
        ys = list(range(self._scales[0]))

        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, _ = self._sample_patches(target_x, target_y, target)

                with tf.GradientTape(persistent=True) as tape:
                    # Perform multi-scale training
                    source_patch, _ = self._sample_patches(x[0], y[0], source)
                    pred, vq = self(source_patch)

                    for i in range(1, len(self._scales) - 1):
                        pred, vq = self._sample_patches(x[i], y[i], pred, vq)

                        if self._config["data"]["type"] == "super_res":
                            pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

                        if self.intermediate_vq:
                            pred, vq = self(vq)
                        else:
                            pred, vq = self(pred)

                    pred, _ = self._sample_patches(x[-1], y[-1], pred)
                    
                    # Calculate L1
                    L1_loss = L1(target_patch, pred)

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
    def test_step(self, source, target):

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self._scales[0]))
        ys = list(range(self._scales[0]))

        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, _ = self._sample_patches(target_x, target_y, target)

                # Perform multi-scale inference
                source_patch, _ = self._sample_patches(x[0], y[0], source)
                pred, vq = self(source_patch)

                for i in range(1, len(self._scales) - 1):
                    pred, vq = self._sample_patches(x[i], y[i], pred, vq)

                    if self._config["data"]["type"] == "super_res":
                        pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

                    if self.intermediate_vq:
                        pred, vq = self(vq)
                    else:
                        pred, vq = self(pred)

                pred, _ = self._sample_patches(x[-1], y[-1], pred)

                # Calculate L1
                L1_loss = L1(target_patch, pred)

                # Calculate VQ loss
                if self.use_vq:
                    vq_loss = sum(self.UNet.losses)
                else:
                    vq_loss = 0

                total_loss = L1_loss + vq_loss
                self.L1_metric.update_state(L1_loss)
                self.vq_metric.update_state(vq_loss)
                self.total_metric.update_state(total_loss)

    def _get_scale_indices(self, target_x=None, target_y=None):
        if target_x is None or target_y is None:
            # Want higher probability of training on more central regions
            if np.random.randn() > 0.5:
                target_x = np.random.randint(0, self._scales[0])
                target_y = np.random.randint(0, self._scales[0])
            else:
                target_x = np.random.randint(
                    self._scales[0] / 4,
                    self._scales[0] - self._scales[0] / 4
                )
                target_y = np.random.randint(
                    self._scales[0] / 4,
                    self._scales[0] - self._scales[0] / 4
                )

        binary_rep = bin(target_x)[2:]
        source_x = [0 for _ in range(len(self._scales) - len(binary_rep))]
        for c in binary_rep:
            source_x.append(int(c))

        binary_rep = bin(target_y)[2:]
        source_y = [0 for _ in range(len(self._scales) - len(binary_rep))]
        for c in binary_rep:
            source_y.append(int(c))

        return source_x, source_y, target_x, target_y

    def _downsample_images(self, img):
        img = img[:, :, ::self._scales[0], ::self._scales[0], :]
        return img

    def _down_sample_super_res(self, img):
        img = img[:, 1::LQ_SLICE_THICK, :, :, :]
        return img

    def _sample_patches(self, x, y, img1, img2=None):
        x_img = x * self._source_dims[1]
        y_img = y * self._source_dims[2]
        img1 = img1[:, :, x_img:(x_img + self._source_dims[1]), y_img:(y_img + self._source_dims[2]), :]

        if img2 is not None:
            img2 = img2[:, :, x_img:(x_img + self._source_dims[1]), y_img:(y_img + self._source_dims[2]), :]

        return img1, img2

    def example_inference(self, source, target):
        x, y, target_x, target_y = self._get_scale_indices(
            self._scales[0] // 2,
            self._scales[0] // 2
        )
        preds = {}
        source_patch, target = self._sample_patches(target_x, target_y, source, target)
        source = self._downsample_images(source)
        source, _ = self._sample_patches(x[0], y[0], source)
        pred, vq = self(source)
        preds[str(self._scales[0])] = pred

        for i in range(1, len(self._scales) - 1):
            pred, vq = self._sample_patches(x[i], y[i], pred, vq)

            if self._config["data"]["type"] == "super_res":
                pred = self._down_sample_super_res(pred) # TODO: handle VQ if necessary

            if self.intermediate_vq:
                pred, vq = self(vq)
                preds[str(self._scales[i])] = vq
            else:
                pred, vq = self(pred)
                preds[str(self._scales[i])] = pred

        pred, vq = self._sample_patches(x[-1], y[-1], pred, vq)
        preds[str(self._scales[-1])] = pred

        return source_patch, target, preds

    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x):
        return self.UNet(x)
