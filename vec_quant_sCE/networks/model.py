import tensorflow as tf

from .components.unet import UNet
from vec_quant_sCE.utils.augmentation import StdAug
from vec_quant_sCE.utils.losses import L1, FocalLoss


#-------------------------------------------------------------------------
""" Wrapper for U-Net """

class Model(tf.keras.Model):

    def __init__(self, config, name="Model"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.mb_size = config["expt"]["mb_size"]
        self.img_dims = config["hyperparameters"]["img_dims"]

        if config["hyperparameters"]["time_layers"] is None:
            self.input_times = False
        else:
            self.input_times = True

        if config["hyperparameters"]["vq_layers"] is None:
            self.use_vq = False
        else:
            self.use_vq = True

        # Set up augmentation
        self.Aug = StdAug(config=config)

        # Check UNet output dims match input
        input_size = [1] + self.img_dims + [1]
        output_size = [1] + self.img_dims + [1]
        self.UNet = UNet(self.initialiser, config["hyperparameters"], name="unet")

        if self.input_times:
            assert self.UNet.build_model(tf.zeros(input_size), tf.zeros(1)) == output_size, f"{self.UNet.build_model(tf.zeros(input_size), tf.zeros(1))} vs {output_size}"
        else:
            assert self.UNet.build_model(tf.zeros(input_size)) == output_size, f"{self.UNet.build_model(tf.zeros(input_size))} vs {output_size}"

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
            outputs = self.UNet.call(source, tf.zeros(1))
        else:
            outputs = self.UNet.call(source)

        print("===========================================================")
        print("UNet")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()

    @tf.function
    def train_step(self, source, target, seg=None, times=None):

        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        # Augmentation if required
        if self.Aug:
            imgs, seg = self.Aug(imgs=[source, target], seg=seg)
            source, target = imgs

        with tf.GradientTape(persistent=True) as tape:

            if self.input_times:
                pred = self.UNet(source, times)
            else:
                pred = self.UNet(source)
            
            # Calculate L1
            if seg is not None:
                L1_loss = self.L1_loss(target, pred, seg)
            else:
                L1_loss = self.L1_loss(target, pred)

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

        # Generate fake target
        if self.input_times:
            pred = self.UNet(source, times)
        else:
            pred = self.UNet(source)

        # Calculate L1
        if seg is not None:
            L1_loss = self.L1_loss(target, pred, seg)
        else:
            L1_loss = self.L1_loss(target, pred)

        if self.use_vq:
            vq_loss = sum(self.UNet.losses)
        else:
            vq_loss = 0
        total_loss = L1_loss + vq_loss
        self.L1_metric.update_state(L1_loss)
        self.vq_metric.update_state(vq_loss)
        self.total_metric.update_state(total_loss)
    
    def reset_train_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def call(self, x, t=None):
        return self.UNet(x, t)
