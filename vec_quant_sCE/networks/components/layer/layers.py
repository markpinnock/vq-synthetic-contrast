import tensorflow as tf


#-------------------------------------------------------------------------
""" Instance normalisation layer for Pix2Pix generator """

class InstanceNorm(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.epsilon = 1e-12
    
    def build(self, input_shape):
        self.beta = self.add_weight("beta", shape=[1, 1, 1, 1, input_shape[-1]], initializer="zeros", trainable=True)
        self.gamma = self.add_weight("gamma", shape=[1, 1, 1, 1, input_shape[-1]], initializer="ones", trainable=True)

    def call(self, x, training=None):
        mu = tf.math.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sigma = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)

        return (x - mu) / (sigma + self.epsilon) * self.gamma + self.beta


#-------------------------------------------------------------------------
""" Down-sampling convolutional block for Pix2pix discriminator and generator """

class GANDownBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - initialiser: e.g. keras.initializers.RandomNormal
        - batch_norm: True/False """

    def __init__(self, nc, weights, strides, initialiser, model, batch_norm=True, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        bias = not batch_norm

        self.conv = tf.keras.layers.Conv3D(nc, weights, strides=strides, padding="same", kernel_initializer=initialiser, use_bias=bias, name="conv")

        # Normalisation
        if batch_norm and model == "generator":
            self.bn = InstanceNorm(name="instancenorm")
        elif batch_norm and model == "discriminator":
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")

        if model == "generator":
            self.time_reps = 1
        else:
            self.time_reps = 2

    def call(self, x, t=None, training=True):
        if t is not None:
            tiled_time = tf.tile(tf.reshape(t, [-1, 1, 1, 1, 1]), [self.time_reps] + x.shape[1:4] + [1], "time_tile")
            x = tf.concat([x, tiled_time], axis=4, name="time_concat")

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x, training=training)

        return tf.nn.leaky_relu(x, alpha=0.2, name="l_relu")


#-------------------------------------------------------------------------
""" Up-sampling convolutional block for Pix2pix generator """

class GANUpBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - initialiser: e.g. keras.initializers.RandomNormal
        - batch_norm: True/False
        - dropout: True/False """

    def __init__(self, nc, weights, strides, initialiser, batch_norm=True, dropout=False, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        bias = not batch_norm
        self.dropout = dropout

        self.tconv = tf.keras.layers.Conv3DTranspose(nc, weights, strides=strides, padding="same", kernel_initializer=initialiser, use_bias=bias, name="tconv")
        self.conv = tf.keras.layers.Conv3D(nc, weights, strides=(1, 1, 1), padding="same", kernel_initializer=initialiser, use_bias=bias, name="conv")

        # Instance normalisation
        if batch_norm:
            self.bn1 = InstanceNorm(name="instancenorm1")
            self.bn2 = InstanceNorm(name="instancenorm2")

        if dropout:
            self.dropout1 = tf.keras.layers.Dropout(0.5, name="dropout1")
            self.dropout2 = tf.keras.layers.Dropout(0.5, name="dropout2")
        
        self.concat = tf.keras.layers.Concatenate(name="concat")
    
    def call(self, x, skip, t=None, training=True):
        if t is not None:
            tiled_time = tf.tile(tf.reshape(t, [-1, 1, 1, 1, 1]), [1] + x.shape[1:4] + [1], "time_tile")
            x = tf.concat([x, tiled_time], axis=4, name="time_concat")

        x = self.tconv(x)

        if self.batch_norm:
            x = self.bn1(x)
        
        if self.dropout:
            x = self.dropout1(x, training=training)
    
        x = tf.nn.relu(x)
        x = self.concat([x, skip])
        x = self.conv(x)

        if self.batch_norm:
            x = self.bn2(x)

        if self.dropout:
            x = self.dropout2(x, training=training)

        return tf.nn.relu(x)
