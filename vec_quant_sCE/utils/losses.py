import tensorflow as tf
import tensorflow.keras as keras


""" Loss functions used in Version 02 - standard GAN losses """

#-------------------------------------------------------------------------
""" Non-saturating minimax loss
    Goodfellow et al. Generative adversarial networks. NeurIPS, 2014
    https://arxiv.org/abs/1406.2661 """

@tf.function
def minimax_D(real_output, fake_output):
    real_loss = keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return 0.5 * tf.reduce_mean(real_loss + fake_loss)

@tf.function
def minimax_G(fake_output):
    fake_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)
    return tf.reduce_mean(fake_loss)


#-------------------------------------------------------------------------
""" Pix2pix L1 loss
    Isola et al. Image-to-image translation with conditional adversarial networks.
    CVPR, 2017.
    https://arxiv.org/abs/1406.2661 """

@tf.function
def L1(real_img, fake_img):
    return tf.reduce_mean(tf.abs(real_img - fake_img), name="L1")


#-------------------------------------------------------------------------
""" Least squares loss
    Mao et al. Least squares generative adversarial networks. ICCV, 2017.
    https://arxiv.org/abs/1611.04076
    
    Zhu et al. Unpaired Image-to-image translation using cycle-consistent
    adversarial networks. ICCV 2017.
    https://arxiv.org/abs/1703.10593 """

@tf.function
def least_squares_D(real_output, fake_output):
    real_loss = 0.5 * tf.reduce_mean(tf.square(real_output - 1))
    fake_loss = 0.5 * tf.reduce_mean(tf.square(fake_output))
    return fake_loss + real_loss

@tf.function
def least_squares_G(fake_output):
    fake_loss = tf.reduce_mean(tf.square(fake_output - 1))
    return fake_loss


#-------------------------------------------------------------------------
""" Gradient difference
    Xie et al. Generation of contrast-enhanced CT with residual cycle-consistent
    generative adversarial networks (Res-CycleGAN).
    Medical Imae 2021: Physics of Medical Imaging. 2021.
    https://doi.org/10.1117/12.2581056 """

@tf.function
def gradient_difference(real_img, fake_img, k):
    real_gradient = tf.nn.conv3d(real_img, filters=k, strides=(1, 1, 1, 1, 1), padding="SAME")
    fake_gradient = tf.nn.conv3d(fake_img, filters=k, strides=(1, 1, 1, 1, 1), padding="SAME")
    gradient_difference = tf.reduce_mean(tf.square(tf.abs(real_gradient) - tf.abs(fake_gradient)))

    return gradient_difference


#-------------------------------------------------------------------------
""" Mutal information
    Xie et al. Generation of contrast-enhanced CT with residual cycle-consistent
    generative adversarial networks (Res-CycleGAN).
    Medical Imae 2021: Physics of Medical Imaging. 2021.
    https://doi.org/10.1117/12.2581056
    
    Implementation adapted from:
    https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/loss/image.py """

@tf.function
def mutual_information(real_img, fake_img):

    # intensity is split into bins between 0, 1
    real_img = tf.clip_by_value(real_img[:, ::4, ::4, ::4, :], 0, 1)
    fake_img = tf.clip_by_value(fake_img[:, ::4, ::4, ::4, :], 0, 1)
    bin_centers = tf.linspace(0.0, 1.0, 16)
    bin_centers = tf.cast(bin_centers, dtype=real_img.dtype)
    bin_centers = bin_centers[None, None, ...]
    sigma = (tf.reduce_mean(bin_centers[:, :, 1:] - bin_centers[:, :, :-1]) * 0.5)
    preterm = 1 / (2 * tf.math.square(sigma))
    batch, w, h, z, c = real_img.shape
    y_true = tf.reshape(real_img, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
    y_pred = tf.reshape(fake_img, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
    nb_voxels = y_true.shape[1] * 1.0  # w * h * z, number of voxels

    # each voxel contributes continuously to a range of histogram bin
    ia = tf.math.exp(-preterm * tf.math.square(y_true - bin_centers))
    ia /= tf.reduce_sum(ia, -1, keepdims=True)
    ia = tf.transpose(ia, (0, 2, 1))
    pa = tf.reduce_mean(ia, axis=-1, keepdims=True)

    ib = tf.math.exp(-preterm * tf.math.square(y_pred - bin_centers))
    ib /= tf.reduce_sum(ib, -1, keepdims=True)
    pb = tf.reduce_mean(ib, axis=1, keepdims=True)

    papb = tf.matmul(pa, pb)
    pab = tf.matmul(ia, ib)
    pab /= nb_voxels

    # MI: sum(P_ab * log(P_ab/P_ap_b))
    div = (pab + 1e-12) / (papb + 1e-12)

    return tf.reduce_sum(pab * tf.math.log(div + 1e-12))

#-------------------------------------------------------------------------
""" Wasserstein loss
    Arjovsky et al. Wasserstein generative adversarial networks.
    International conference on machine learning. PMLR, 2017
    https://arxiv.org/abs/1701.07875 """

@tf.function
def wasserstein_D(real_output, fake_output):
    return tf.reduce_mean(fake_output - real_output)

@tf.function
def wasserstein_G(fake_output):
    return tf.reduce_mean(-fake_output)


#-------------------------------------------------------------------------
""" Wasserstein loss gradient penalty
    Gulrajani et al. Improved training of Wasserstein GANs. NeurIPS, 2017
    https://arxiv.org/abs/1704.00028 """

@tf.function
def gradient_penalty(real_img, fake_img, D, scale):
    epsilon = tf.random.uniform([fake_img.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = (epsilon * real_img) + ((1 - epsilon) * fake_img)

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        D_hat = D(x_hat, scale, training=True)
    
    gradients = tape.gradient(D_hat, x_hat)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=(1, 2)) + 1e-8)
    grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

    return grad_penalty


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


#-------------------------------------------------------------------------
""" Focal metric, weights loss according to masked area """

class FocalMetric(keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.global_loss = self.add_weight(name="global", initializer="zeros")
        self.focal_loss = self.add_weight(name="focal", initializer="zeros")
        self.N = self.add_weight(name="N", initializer="zeros")
        self.loss = focused_mae

    def update_state(self, y, x, mask):
        global_loss, focal_loss = self.loss(x, y, mask)
        self.global_loss.assign_add(global_loss)
        self.focal_loss.assign_add(focal_loss)
        self.N.assign_add(x.shape[0])
    
    def result(self):
        return [self.global_loss / self.N, self.focal_loss / self.N]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)
        self.N.assign(1e-12) # So no nans if result called after reset_states
