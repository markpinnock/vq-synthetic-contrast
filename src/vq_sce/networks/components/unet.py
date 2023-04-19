from typing import Any, TypedDict

import numpy as np
import tensorflow as tf

from .layers.conv_layers import BottomBlock, DownBlock, UpBlock, VQConfigType
from .layers.vq_layers import VQBlock

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


class UNet(tf.keras.Model):
    _vq_config: VQConfigType | None
    _vq_layers: list[str] | None

    def __init__(
        self,
        initialiser: tf.keras.initializers.Initializer,
        config: dict[str, Any],
        shared_vq: VQBlock | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Check network and image dimensions
        self._source_dims = tuple(config["source_dims"])
        self._target_dims = tuple(config["target_dims"])
        assert len(self._source_dims) == 3, "3D input only"
        self._config = config
        self._z_upsamp_factor = self._target_dims[0] // self._source_dims[0]

        if config["vq_layers"] is not None:
            self._vq_layers = config["vq_layers"].keys()
            self._vq_config = {"vq_beta": config["vq_beta"]}  # type: ignore[assignment]
        else:
            self._vq_layers = []
            self._vq_config = None

        self._initialiser = initialiser
        max_num_layers = int(
            np.log2(np.min([self._source_dims[1], self._source_dims[2]])),
        )
        assert (
            config["layers"] <= max_num_layers and config["layers"] >= 0
        ), f"Maximum number of generator layers: {max_num_layers}"

        self.shared_vq = shared_vq
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

            use_vq = f"down_{i}" in self._vq_layers
            if use_vq:
                self._vq_config["embeddings"] = self._config["vq_layers"][f"down_{i}"]

            self.encoder.append(
                DownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=self._initialiser,
                    use_vq=use_vq,
                    vq_config=self._vq_config,
                    name=f"down_{i}",
                ),
            )

        use_vq = "bottom" in self._vq_layers
        if use_vq:
            self._vq_config["embeddings"] = self._config["vq_layers"]["bottom"]

        self.bottom_layer = BottomBlock(
            channels,
            kernel,
            (1, 1, 1),
            initialiser=self._initialiser,
            use_vq=use_vq,
            vq_config=self._vq_config,
            shared_vq=self.shared_vq,
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

            use_vq = f"up_{i}" in self._vq_layers
            if use_vq:
                self._vq_config["embeddings"] = self._config["vq_layers"][f"up_{i}"]

            self.decoder.append(
                UpBlock(
                    channels,
                    kernel,
                    strides,
                    upsamp_factor=upsamp_factor,
                    initialiser=self._initialiser,
                    use_vq=use_vq,
                    vq_config=self._vq_config,
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
        )

        if "final" in self._vq_layers:
            self.output_vq = VQBlock(
                num_embeddings=self._vq_config["embeddings"],
                embedding_dim=1,
                beta=self._vq_config["vq_beta"],
                name="output_vq",
            )
        else:
            self.output_vq = None

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor | None]:
        skip_layers = []

        if self._z_upsamp_factor > 1:
            residual_x = self.upsample(x)
        else:
            residual_x = x

        for layer in self.encoder:
            x, skip = layer(x, training=True)
            skip_layers.append(skip)

        x = self.bottom_layer(x, training=True)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            x = tconv(x, skip, training=True)

        x = self.final_layer(x, training=True)

        if self.output_vq is None:
            return x + residual_x, None

        else:
            return x + residual_x, self.output_vq(x) + residual_x


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

        use_vq = "upsamp" in self._vq_layers
        if use_vq:
            self._vq_config["embeddings"] = self._config["vq_layers"]["upsamp"]

        self.upsample_out = UpBlock(
            cache["channels"][0],
            (2, 4, 4),
            (1, 2, 2),
            upsamp_factor=1,
            initialiser=self._initialiser,
            use_vq=use_vq,
            vq_config=self._vq_config,
            name="upsamp",
        )

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor | None]:
        skip_layers = []
        residual_x = self.upsample(x)

        for layer in self.encoder:
            x, skip = layer(x, training=True)
            skip_layers.append(skip)

        x = self.bottom_layer(x, training=True)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            x = tconv(x, skip, training=True)

        x = self.upsample_out(x, residual_x)
        x = self.final_layer(x, training=True)

        if self.output_vq is None:
            return x + residual_x, None

        else:
            return x + residual_x, self.output_vq(x) + residual_x
