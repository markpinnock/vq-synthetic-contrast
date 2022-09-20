import tensorflow as tf
import tensorflow.keras as keras


#-------------------------------------------------------------------------
""" Pix2pix L1 loss
    Isola et al. Image-to-image translation with conditional adversarial networks.
    CVPR, 2017.
    https://arxiv.org/abs/1406.2661 """

@tf.function
def L1(real_img, fake_img):
    return tf.reduce_mean(tf.abs(real_img - fake_img), name="L1")


#-------------------------------------------------------------------------
""" Focused L1 loss, calculates L1 inside and outside masked area """

@tf.function
def focused_mae(x, y, m):
    global_absolute_err = tf.abs(x - y)
    focal_absolute_err = tf.abs(x - y) * m
    global_mae = tf.reduce_mean(global_absolute_err)
    focal_mae = tf.reduce_sum(focal_absolute_err) / (tf.reduce_sum(m) + 1e-12)

    return global_mae, focal_mae


#-------------------------------------------------------------------------
""" Focal loss, weights loss according to masked area """

class FocalLoss(keras.layers.Layer):
    def __init__(self, mu, name=None):
        super().__init__(name=name)
        assert mu <= 1.0 and mu >= 0.0, "Mu must be in range [0, 1]"
        self.mu = mu
        self.loss = focused_mae

    def call(self, y, x, mask):
        global_loss, focal_loss = self.loss(x, y, mask)

        return (1 - self.mu) * global_loss + self.mu * focal_loss
