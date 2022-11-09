import numpy as np
import tensorflow as tf


#-------------------------------------------------------------------------

class UNet(tf.keras.Model):
    def __init__(self, config, name="UNet"):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 2, "2D input only"
        num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]]))) - 2

        self.encoder = []

        # Cache channels, strides and weights
        channel_cache = []

        for i in range(0, num_layers - 1):
            channels = np.min([config["nc"] * 2 ** i, 512])
            channel_cache.append(channels)
            self.encoder.append(DownBlock(channels, name=f"down_{i}"))

        self.bottom_layer = DownBlock(channels, name="bottom")
        channel_cache.reverse()
        self.decoder = []

        for i in range(0, num_layers - 1):
            channels = channel_cache[i]
            self.decoder.append(UpBlock(channels,name=f"up_{i}"))

        self.final_layer = tf.keras.layers.Conv2DTranspose(1, (3, 3), (2, 2), padding="same", activation="sigmoid", kernel_initializer="he_normal", name="output", use_bias=True)

    def call(self, x, training):
        skip_layers = []

        for block in self.encoder:
            x = block(x, training=training)
            skip_layers.append(x)

        x = self.bottom_layer(x, training=training)
        skip_layers.reverse()

        for skip, block in zip(skip_layers, self.decoder):
            x = block(x, skip, training=training)

        x = self.final_layer(x)

        return x


#-------------------------------------------------------------------------
""" Down-sampling convolutional block for U-Net"""

class DownBlock(tf.keras.layers.Layer):

    def __init__(self, nc, name):
        super().__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(nc, (3, 3), strides=(1, 1), padding="same", activation="linear", kernel_initializer="he_normal", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(nc, (3, 3), strides=(1, 1), padding="same", activation="linear", kernel_initializer="he_normal", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same")

    def call(self, x, training):

        x = tf.nn.relu(self.bn1(self.conv1(x), training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training))

        return self.pool(x)


#-------------------------------------------------------------------------
""" Up-sampling convolutional block for U-Net"""

class UpBlock(tf.keras.layers.Layer):
    
    def __init__(self, nc, name):
        super().__init__(name=name)

        self.tconv = tf.keras.layers.Conv2DTranspose(nc, (2, 2), strides=(2, 2), padding="same", activation="linear", kernel_initializer="he_normal", use_bias=False)
        self.tbn = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(nc, (3, 3), strides=(1, 1), padding="same", activation="linear", kernel_initializer="he_normal", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(nc, (3, 3), strides=(1, 1), padding="same", activation="linear", kernel_initializer="he_normal", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, skip, training):

        x = tf.nn.relu(self.tbn(self.tconv(x), training))
        x = tf.keras.layers.concatenate([x, skip], axis=3)
        x = tf.nn.relu(self.bn1(self.conv1(x), training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training))

        return x