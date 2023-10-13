from typing import Any, TypedDict

import numpy as np
import tensorflow as tf

from .layers.conv_layers import DownBlock, UpBlock
from .layers.vq_layers import DARTSVQBlock, VQBlock

MAX_CHANNELS = 512


# -------------------------------------------------------------------------


class CacheType(TypedDict):
    channels: list[int]
    encode_strides: list[tuple[int, int, int]]
    encode_kernels: list[tuple[int, int, int]]
    decode_strides: list[tuple[int, int, int]]
    decode_kernels: list[tuple[int, int, int]]
    upsamp_factor: list[int]


# -------------------------------------------------------------------------


class UNet(tf.keras.layers.Layer):
    def __init__(
        self,
        initialiser: tf.keras.initializers.Initializer,
        config: dict[str, Any],
        vq_blocks: dict[str, VQBlock | DARTSVQBlock | None],
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        try:
            self._residual = config["residual"]
        except KeyError:
            self._residual = True

        # Check network and image dimensions
        self._source_dims = tuple(config["source_dims"])
        self._target_dims = tuple(config["target_dims"])
        assert len(self._source_dims) == 3, "3D input only"
        self._config = config
        self._z_upsamp_factor = self._target_dims[0] // self._source_dims[0]

        self._initialiser = initialiser
        max_num_layers = int(
            np.log2(np.min([self._source_dims[1], self._source_dims[2]])),
        )
        assert (
            config["layers"] <= max_num_layers and config["layers"] >= 0
        ), f"Maximum number of generator layers: {max_num_layers}"

        self.vq_blocks = vq_blocks
        self.encoder: list[tf.keras.layers.Layer] = []
        self.decoder: list[tf.keras.layers.Layer] = []
        cache = self.get_encoder()
        self.get_decoder(cache)

    def get_encoder(self) -> CacheType:
        """Create U-Net encoder."""
        # Cache channels, strides and weights
        cache: CacheType = {
            "channels": [],
            "encode_strides": [],
            "encode_kernels": [],
            "decode_strides": [],
            "decode_kernels": [],
            "upsamp_factor": [],
        }

        # Upsample in z-direction for residual if needed
        self.upsample = tf.keras.layers.UpSampling3D(size=(self._z_upsamp_factor, 1, 1))

        # Determine number of up-scaling layers needed
        source_z = self._source_dims[0]
        target_z = self._target_dims[0]

        for i in range(0, self._config["layers"]):
            channels = np.min([self._config["nc"] * 2**i, MAX_CHANNELS])
            cache["channels"].append(channels)
            cache["upsamp_factor"].append(target_z // source_z)

            # If source z dim is odd or if source feature dim
            # smaller than target feature dim, don't downsample source
            if (source_z / 2) - (source_z // 2) != 0 or source_z < target_z:
                cache["encode_strides"].append((1, 2, 2))
                cache["encode_kernels"].append((2, 4, 4))
            else:
                cache["encode_strides"].append((2, 2, 2))
                cache["encode_kernels"].append((4, 4, 4))
                source_z //= 2

            # If target z dim will be odd, don't upsample
            if (target_z / 2) - (target_z // 2) != 0:
                cache["decode_strides"].append((1, 2, 2))
                cache["decode_kernels"].append((2, 4, 4))
            else:
                cache["decode_strides"].append((2, 2, 2))
                cache["decode_kernels"].append((4, 4, 4))
                target_z //= 2

        for i in range(0, self._config["layers"]):
            channels = cache["channels"][i]
            strides = cache["encode_strides"][i]
            kernel = cache["encode_kernels"][i]

            self.encoder.append(
                DownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=self._initialiser,
                    name=f"down_{i}",
                ),
            )

        self.bottom_layer = DownBlock(
            channels,
            kernel,
            (1, 1, 1),
            initialiser=self._initialiser,
            name="bottom",
        )

        return cache

    def get_decoder(self, cache: CacheType) -> None:
        """Create U-Net decoder."""
        for i in range(self._config["layers"] - 1, -1, -1):
            channels = cache["channels"][i]
            strides = cache["decode_strides"][i]
            kernel = cache["decode_kernels"][i]
            upsamp_factor = cache["upsamp_factor"][i]

            self.decoder.append(
                UpBlock(
                    channels,
                    kernel,
                    strides,
                    upsamp_factor=upsamp_factor,
                    initialiser=self._initialiser,
                    name=f"up_{i}",
                ),
            )

        self.final_layer = tf.keras.layers.Conv3D(
            1,
            (1, 1, 1),
            (1, 1, 1),
            padding="same",
            activation="tanh",
            kernel_initializer=self._initialiser,
            name="final",
            dtype="float32",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        skip_layers = []
        residual_x = self.upsample(x)
        residual_x = tf.cast(residual_x, "float32")

        for layer in self.encoder:
            x, skip = layer(x, training=True)
            skip_layers.append(skip)

        x, _ = self.bottom_layer(x, training=True)
        skip_layers.reverse()

        if "bottom" in self.vq_blocks.keys():
            x = self.vq_blocks["bottom"](x)

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.vq_blocks.keys():
                skip = self.vq_blocks[tconv.name](skip)
            x = tconv(x, skip, training=True)

        x = self.final_layer(x, training=True)

        if self._residual:
            return x + residual_x
        else:
            return x


# -------------------------------------------------------------------------


class MultiscaleUNet(UNet):
    """Multiscale UNet.
    :param initialiser: e.g. keras.initializers.RandomNormal
    :param nc: number of channels in first layer
    :param num_layers: number of layers
    :param img_dims: input image size
    """

    def __init__(
        self,
        initialiser: tf.keras.initializers.Initializer,
        config: dict[str, Any],
        shared_vq: VQBlock | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(initialiser, config, shared_vq, name=name)

    def get_encoder(self) -> CacheType:
        """Create multi-scale U-Net encoder."""
        cache = super().get_encoder()

        # Upsample in z-direction for residual if needed
        self.upsample = tf.keras.layers.UpSampling3D(size=(self._z_upsamp_factor, 2, 2))

        return cache

    def get_decoder(self, cache: CacheType) -> None:
        """Create multi-scale U-Net decoder."""
        super().get_decoder(cache)

        self.upsample_out = UpBlock(
            cache["channels"][0],
            (2, 4, 4),
            (1, 2, 2),
            upsamp_factor=1,
            initialiser=self._initialiser,
            name="upsamp",
        )

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor | None]:
        skip_layers = []
        residual_x = self.upsample(x)
        residual_x = tf.cast(residual_x, "float32")

        for layer in self.encoder:
            x, skip = layer(x, training=True)
            skip_layers.append(skip)

        x, _ = self.bottom_layer(x, training=True)
        skip_layers.reverse()

        if "bottom" in self.vq_blocks.keys():
            x = self.vq_blocks["bottom"](x)

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.vq_blocks.keys():
                skip = self.vq_blocks[tconv.name](skip)
            x = tconv(x, skip, training=True)

        x = self.upsample_out(x, residual_x)
        x = self.final_layer(x, training=True)

        if self._residual:
            return x + residual_x, None
        else:
            return x, None
