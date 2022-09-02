import tensorflow as tf

from .components.generator import Generator
from vec_quant_sCE.utils.augmentation import StdAug
from vec_quant_sCE.utils.losses import L1, FocalLoss, FocalMetric


#-------------------------------------------------------------------------
""" Wrapper for U-Net """

class UNet(tf.keras.Model):

    def __init__(self, config, name="UNet"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.mb_size = config["expt"]["mb_size"]
        self.img_dims = config["hyperparameters"]["img_dims"]

        if config["hyperparameters"]["g_time_layers"] is None:
            self.input_times = False
        else:
            self.input_times = True

        # Set up augmentation
        self.Aug = StdAug(config=config)

        # Check UNet output dims match input
        input_size = [1] + self.img_dims + [1]
        output_size = [1] + self.img_dims + [1]
        self.UNet = Generator(self.initialiser, config["hyperparameters"], mode="UNet", name="generator")

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
        if len(self.config["data"]["segs"]) > 0:
            self.train_L1_metric = FocalMetric(name="train_L1")
            self.val_L1_metric = FocalMetric(name="val_L1")

        else:
            self.train_L1_metric = tf.keras.metrics.Mean(name="train_L1")
            self.val_L1_metric = tf.keras.metrics.Mean(name="val_L1")

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
    def train_step(self, real_source, real_target, seg=None, source_times=None, target_times=None):

        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        # Augmentation if required
        if self.Aug:
            imgs, seg = self.Aug(imgs=[real_source, real_target], seg=seg)
            real_source, real_target = imgs

        with tf.GradientTape(persistent=True) as tape:

            # Generate fake target
            if self.input_times:
                fake_target = self.UNet(real_source, target_times)
            else:
                fake_target = self.UNet(real_source)
            
            # Calculate L1
            if seg is not None:
                loss = self.L1_loss(real_target, fake_target, seg)
                self.train_L1_metric.update_state(real_target, fake_target, seg)

            else:
                loss = self.L1_loss(real_target, fake_target)
                self.train_L1_metric.update_state(loss)

        # Get gradients and update weights
        grads = tape.gradient(loss, self.UNet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.UNet.trainable_variables))

    @tf.function
    def test_step(self, real_source, real_target, seg=None, source_times=None, target_times=None):

        # Generate fake target
        if self.input_times:
            fake_target = self.UNet(real_source, target_times)
        else:
            fake_target = self.UNet(real_source)

        val_L1 = L1(real_target, fake_target)

        if seg is not None:
            self.val_L1_metric.update_state(real_target, fake_target, seg)
        else:
            self.val_L1_metric.update_state(val_L1)
    
    def reset_train_metrics(self):
        self.train_L1_metric.reset_states()

    def call(self, x, t):
        return self.UNet(x, t)
