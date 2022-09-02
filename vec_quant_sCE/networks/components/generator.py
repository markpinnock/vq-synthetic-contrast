import numpy as np
import tensorflow as tf

from .layer.layers import GANDownBlock, GANUpBlock


""" Generator for Pix2pix (can be used for UNet with mode=="UNet") """

class Generator(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, mode="GAN", name=None):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ngf = config["ngf"]
        num_layers = config["g_layers"]
        
        if config["g_time_layers"] is not None:
            self.time_layers = config["g_time_layers"]
        else:
           self.time_layers = []

        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum number of generator layers: {max_num_layers}"
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([ngf * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            self.encoder.append(
                GANDownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    model="generator",
                    batch_norm=True, name=f"down_{i}"))

        self.bottom_layer = GANDownBlock(
            channels,
            kernel,
            strides,
            initialiser=initialiser,
            model="generator",
            batch_norm=True, name="bottom")

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []

        # If mode == UNet, dropout is switched off, else dropout used as in Pix2Pix
        if mode == "GAN":
            dropout = True
        elif mode == "UNet":
            dropout = False
        else:
            raise ValueError

        for i in range(0, num_layers - 1):
            if i > 2: dropout = False
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]

            self.decoder.append(
                GANUpBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    batch_norm=True,
                    dropout=dropout, name=f"up_{i}"))

        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding="same", activation="linear",
            kernel_initializer=initialiser, name="output")
        
        layer_names = [layer.name for layer in self.encoder] + ["bottom"] + [layer.name for layer in self.decoder]

        for time_input in self.time_layers:
            assert time_input in layer_names, (time_input, layer_names)

    def build_model(self, x, t=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, t).shape

    def call(self, x, t=None):
        skip_layers = []

        for conv in self.encoder:
            if conv.name in self.time_layers:
                x = conv(x, t, training=True)
            else:
                x = conv(x, training=True)

            skip_layers.append(x)

        if self.bottom_layer.name in self.time_layers:
            x = self.bottom_layer(x, t, training=True)
        else:
            x = self.bottom_layer(x, training=True)

        x = tf.nn.relu(x)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.time_layers:
                x = tconv(x, skip, t, training=True)
            else:
                x = tconv(x, skip, training=True)

        if self.final_layer.name in self.time_layers:

            x = self.final_layer(x, t, training=True)
        else:
            x = self.final_layer(x, training=True)

        return x