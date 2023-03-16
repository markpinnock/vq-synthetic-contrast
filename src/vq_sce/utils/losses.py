import tensorflow as tf
import tensorflow.keras as keras


""" Pix2pix L1 loss
    Isola et al. Image-to-image translation with conditional adversarial networks.
    CVPR, 2017.
    https://arxiv.org/abs/1406.2661 """

@tf.function
def L1(real_img: tf.Tensor, fake_img: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(real_img - fake_img), name="L1")
