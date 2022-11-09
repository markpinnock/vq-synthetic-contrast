import numpy as np
import tensorflow as tf

from .components.unet import UNet
from vec_quant_sCE.utils.augmentation import StdAug
from vec_quant_sCE.utils.losses import L1, FocalLoss


#-------------------------------------------------------------------------
""" Wrapper for multi-scale U-Net """

class MultiscaleModel(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.mb_size = config["expt"]["mb_size"]
        self.img_dims = config["data"]["patch_size"]
        config["hyperparameters"]["img_dims"] = self.img_dims
        self.intermediate_vq = "output" in config["hyperparameters"]["vq_layers"]
        self.scales = config["hyperparameters"]["scales"]

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

        if self.config["hyperparameters"]["mu"] > 0.0:
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

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self.scales[0]))
        ys = list(range(self.scales[0]))
        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, seg_patch = self._sample_patches(target_x, target_y, target, seg)

                with tf.GradientTape(persistent=True) as tape:
                    # Perform multi-scale training
                    source_patch, _ = self._sample_patches(x[0], y[0], source)
                    pred, vq = self(source_patch, times)

                    for i in range(1, len(self.scales) - 1):
                        pred, vq = self._sample_patches(x[i], y[i], pred, vq)
                        if self.intermediate_vq:
                            pred, vq = self(vq, times)
                        else:
                            pred, vq = self(pred, times)

                    pred, _ = self._sample_patches(x[-1], y[-1], pred)
                    
                    # Calculate L1
                    if seg is not None:
                        L1_loss = self.L1_loss(target_patch, pred, seg_patch)
                    else:
                        L1_loss = self.L1_loss(target_patch, pred)

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

        # Down-sample source image
        source = self._downsample_images(source)

        # Iterate over image patches, get patch indices for each scale
        xs = list(range(self.scales[0]))
        ys = list(range(self.scales[0]))
        for target_x in xs:
            for target_y in ys:
                x, y, target_x, target_y = self._get_scale_indices(target_x, target_y)
                target_patch, seg_patch = self._sample_patches(target_x, target_y, target, seg)

                # Perform multi-scale inference
                source_patch, _ = self._sample_patches(x[0], y[0], source)
                pred, vq = self(source_patch, times)

                for i in range(1, len(self.scales) - 1):
                    pred, vq = self._sample_patches(x[i], y[i], pred, vq)
                    if self.intermediate_vq:
                        pred, vq = self(vq, times)
                    else:
                        pred, vq = self(pred, times)

                pred, _ = self._sample_patches(x[-1], y[-1], pred)

                # Calculate L1
                if seg is not None:
                    L1_loss = self.L1_loss(target_patch, pred, seg_patch)
                else:
                    L1_loss = self.L1_loss(target_patch, pred)

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
                target_x = np.random.randint(0, self.scales[0])
                target_y = np.random.randint(0, self.scales[0])
            else:
                target_x = np.random.randint(self.scales[0] / 4, self.scales[0] - self.scales[0] / 4)
                target_y = np.random.randint(self.scales[0] / 4, self.scales[0] - self.scales[0] / 4)

        binary_rep = bin(target_x)[2:]
        source_x = [0 for _ in range(len(self.scales) - len(binary_rep))]
        for c in binary_rep:
            source_x.append(int(c))

        binary_rep = bin(target_y)[2:]
        source_y = [0 for _ in range(len(self.scales) - len(binary_rep))]
        for c in binary_rep:
            source_y.append(int(c))

        return source_x, source_y, target_x, target_y

    def _downsample_images(self, img):
        img = img[:, ::self.scales[0], ::self.scales[0], :, :]
        return img

    def _sample_patches(self, x, y, img1, img2=None):
        x_img = x * self.img_dims[0]
        y_img = y * self.img_dims[1]
        img1 = img1[:, x_img:(x_img + self.img_dims[0]), y_img:(y_img + self.img_dims[1]), :, :]

        if img2 is not None:
            img2 = img2[:, x_img:(x_img + self.img_dims[0]), y_img:(y_img + self.img_dims[1]), :, :]

        return img1, img2

    def example_inference(self, source, target, seg=None, times=None):
        x, y, target_x, target_y = self._get_scale_indices(
            self.scales[0] // 2,
            self.scales[0] // 2
        )
        preds = {}
        source_patch, target = self._sample_patches(target_x, target_y, source, target)
        source = self._downsample_images(source)
        source, _ = self._sample_patches(x[0], y[0], source)
        pred, vq = self(source, times)
        preds[str(self.scales[0])] = pred

        for i in range(1, len(self.scales) - 1):
            pred, vq = self._sample_patches(x[i], y[i], pred, vq)
            if self.intermediate_vq:
                pred, vq = self(vq, times)
                preds[str(self.scales[i])] = vq
            else:
                pred, vq = self(pred, times)
                preds[str(self.scales[i])] = pred

        pred, vq = self._sample_patches(x[-1], y[-1], pred, vq)
        preds[str(self.scales[-1])] = pred

        return source_patch, target, preds

    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x, t=None):
        return self.UNet(x, t)
