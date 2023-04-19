from typing import TypedDict

import tensorflow as tf

from .vq_layers import VQBlock

# -------------------------------------------------------------------------


class VQConfigType(TypedDict):
    vq_beta: float
    embeddings: int


# -------------------------------------------------------------------------


class InstanceNorm(tf.keras.layers.Layer):
    """Instance normalisation layer for Pix2Pix generator."""

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self.epsilon = 1e-12

    def build(self, input_shape: list[int]) -> None:
        self.beta = self.add_weight(
            "beta",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="zeros",
            trainable=True,
        )
        self.gamma = self.add_weight(
            "gamma",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="ones",
            trainable=True,
        )

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        mu = tf.math.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sigma = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)

        return (x - mu) / (sigma + self.epsilon) * self.gamma + self.beta


# -------------------------------------------------------------------------


class DownBlock(tf.keras.layers.Layer):
    """Down-sampling convolutional block."""

    def __init__(
        self,
        nc: int,
        weights: tuple[int, int, int],
        strides: tuple[int, int, int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: VQConfigType | None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv1",
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv2",
        )
        self.use_vq = use_vq
        if use_vq and vq_config is not None:
            self.vq = VQBlock(
                vq_config["embeddings"],
                nc,
                vq_config["vq_beta"],
                name=f"{name}_vq",
            )

        # Normalisation
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

    def call(self, x: tf.Tensor, training: bool) -> tuple[tf.Tensor, tf.Tensor]:
        # 1st convolution - output to skip layer
        x = self.conv1(x)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)
        skip = x

        # 2nd convolution and down-sample
        x = self.conv2(x)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x), skip


# -------------------------------------------------------------------------


class BottomBlock(tf.keras.layers.Layer):
    """Bottom convolutional block."""

    def __init__(
        self,
        nc: int,
        weights: tuple[int, int, int],
        strides: tuple[int, int, int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: VQConfigType | None,
        shared_vq: VQBlock | None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv1",
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv2",
        )
        self.use_vq = use_vq
        if use_vq and shared_vq is None and vq_config is not None:
            self.vq = VQBlock(
                vq_config["embeddings"],
                nc,
                vq_config["vq_beta"],
                name=f"{name}_vq",
            )
        elif use_vq and shared_vq is not None and vq_config is not None:
            self.vq = shared_vq
        else:
            pass

        # Normalisation
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        # 1st convolution
        x = self.conv1(x)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)

        # 2nd convolution
        x = self.conv2(x)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x)


# -------------------------------------------------------------------------


class UpBlock(tf.keras.layers.Layer):
    """Up-sampling convolutional block."""

    def __init__(
        self,
        nc: int,
        weights: tuple[int, int, int],
        strides: tuple[int, int, int],
        upsamp_factor: int,
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: VQConfigType | None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._upsamp_factor = upsamp_factor
        self.tconv = tf.keras.layers.Conv3DTranspose(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="tconv",
        )
        self.conv1 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv1",
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc,
            weights,
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv2",
        )

        self.use_vq = use_vq
        if use_vq and vq_config is not None:
            self.vq = VQBlock(
                vq_config["embeddings"],
                nc,
                vq_config["vq_beta"],
                name=f"{name}_vq",
            )

        # Instance normalisation
        self.inst_norm_t = InstanceNorm(name="instancenormt")
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

        self.concat = tf.keras.layers.Concatenate(name="concat")

        # Up-sampling module for skip layer if differing input and output dims
        if upsamp_factor > 1:
            self.upsample_skip = UpsampleSkip(
                nc,
                (upsamp_factor * 2, 1, 1),
                strides=(upsamp_factor, 1, 1),
                initialiser=initialiser,
                name="upsample_skip",
            )

    def call(self, x: tf.Tensor, skip: tf.Tensor, training: bool) -> tf.Tensor:
        # Transpose convolution and up-sample
        x = self.tconv(x)
        x = self.inst_norm_t(x, training)
        x = tf.nn.relu(x)

        # Upsample skip if needed
        if self._upsamp_factor > 1:
            skip = self.upsample_skip(skip, training)

        # 1st and 2nd convolutions
        x = self.concat([x, skip])
        x = self.conv1(x)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x)


# -------------------------------------------------------------------------


class UpsampleSkip(tf.keras.layers.Layer):
    """Layer for up-sampling skip layer if different input and output depth."""

    def __init__(
        self,
        nc: int,
        weights: tuple[int, int, int],
        strides: tuple[int, int, int],
        initialiser: tf.keras.initializers.Initializer,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.tconv = tf.keras.layers.Conv3DTranspose(
            nc,
            weights,
            strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="tconv",
        )

        self.inst_norm = InstanceNorm(name="instancenorm")

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.tconv(x)
        x = self.inst_norm(x, training)

        return tf.nn.relu(x)
