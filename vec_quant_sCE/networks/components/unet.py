import numpy as np
import tensorflow as tf

from .layer.layers import DownBlock, UpBlock, UpBlockNoSkip, VQBlock


class UNet(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        
        if config["time_layers"] is not None:
            self.time_layers = config["time_layers"]
        else:
           self.time_layers = []

        if config["vq_layers"] is not None:
            self.vq_layers = config["vq_layers"]
            vq_config = {
                "vq_embeddings": config["vq_embeddings"],
                "vq_time": config["vq_time"],
                "vq_beta": config["vq_beta"]
            }
        else:
           self.vq_layers = []
           vq_config = None

        assert config["layers"] <= max_num_layers and config["layers"] >= 0, f"Maximum number of generator layers: {max_num_layers}"
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, config["layers"] - 1):
            channels = np.min([config["nf"] * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            use_vq = f"down_{i}" in self.vq_layers
            self.encoder.append(
                DownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    use_vq=use_vq,
                    vq_config=vq_config,
                    name=f"down_{i}")
                )

        use_vq = f"bottom" in self.vq_layers
        self.bottom_layer = DownBlock(
            channels,
            kernel,
            strides,
            initialiser=initialiser,
            use_vq=use_vq,
            vq_config=vq_config,
            name="bottom"
        )

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []

        for i in range(0, config["layers"] - 1):
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]

            use_vq = f"up_{i}" in self.vq_layers
            self.decoder.append(
                UpBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    use_vq=use_vq,
                    vq_config=vq_config,
                    name=f"up_{i}")
                )

        use_vq = f"up_{i + 1}" in self.vq_layers
        if config["upsample_layer"]:
            self.upsample_layer = self.decoder.append(
                UpBlockNoSkip(
                    channels,
                    (4, 4, 2),
                    (2, 2, 1),
                    initialiser=initialiser,
                    use_vq=use_vq,
                    vq_config=vq_config,
                    name=f"up_{i + 1}")
                )

        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding="same", activation="linear",
            kernel_initializer=initialiser, name="output")

        if "output" in self.vq_layers:
            self.output_vq = VQBlock(vq_config["vq_embeddings"], 1, vq_config["vq_beta"], name="output_vq")
        else:
            self.output_vq = None
        
        layer_names = [layer.name for layer in self.encoder] + ["bottom"] + [layer.name for layer in self.decoder]

        for time_input in self.time_layers:
            assert time_input in layer_names, (time_input, layer_names)

    def build_model(self, x, t=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, t)

    def call(self, x, t=None):
        skip_layers = []

        for layer in self.encoder:
            if layer.name in self.time_layers:
                x = layer(x, t, training=True)
            else:
                x = layer(x, training=True)
            print(x.shape)
            skip_layers.append(x)

        if self.bottom_layer.name in self.time_layers:
            x = self.bottom_layer(x, t, training=True)
        else:
            x = self.bottom_layer(x, training=True)
        print(x.shape)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.time_layers:
                x = tconv(x, skip, t, training=True)
            else:
                x = tconv(x, skip, training=True)
            print(x.shape)
        if len(self.decoder) > len(self.encoder):
            if self.decoder[-1].name in self.time_layers:
                x = self.decoder[-1](x, t, training=True)
            else:
                x = self.decoder[-1](x, training=True)
        print(x.shape)
        if self.final_layer.name in self.time_layers:
            x = self.final_layer(x, t, training=True)
        else:
            x = self.final_layer(x, training=True)
        print(x.shape)
        if self.output_vq is None:
            return x, None
        else:
            return x, self.output_vq(x)
