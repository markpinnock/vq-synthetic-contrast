import tensorflow as tf

WEIGHT_SCALE = 0.02


#-------------------------------------------------------------------------
""" Instance normalisation layer for Pix2Pix generator """

class InstanceNorm(tf.keras.layers.Layer):

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self.epsilon = 1e-12
    
    def build(self, input_shape: list[int]) -> None:
        self.beta = self.add_weight(
            "beta",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="zeros",
            trainable=True
        )
        self.gamma = self.add_weight(
            "gamma",
            shape=[1, 1, 1, 1, input_shape[-1]],
            initializer="ones",
            trainable=True
        )

    def call(self, x, **kwargs):
        mu = tf.math.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        sigma = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)

        return (x - mu) / (sigma + self.epsilon) * self.gamma + self.beta


#-------------------------------------------------------------------------
""" Down-sampling convolutional block """

class DownBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        nc: int,
        weights: tuple[int],
        strides: tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: dict,
        name: str | None = None
    ) -> None:

        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv3D(
            nc, weights, strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv1"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc, weights, strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv2"
        )
        self.use_vq = use_vq
        if use_vq:
            self.vq = VQBlock(
                vq_config["embeddings"],
                nc, vq_config["vq_beta"],
                name=f"{name}_vq"
            )

        # Normalisation
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

    def call(self, x: tf.Tensor, training: bool) -> tuple[tf.Tensor]:

        # 1st convolution - output to skip layer
        x = self.conv1(x, training)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)
        skip = x

        # 2nd convolution and down-sample
        x = self.conv2(x, training)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x), skip


#-------------------------------------------------------------------------
""" Bottom convolutional block """

class BottomBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        nc: int,
        weights: tuple[int],
        strides: tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: dict,
        name: str | None = None
    ) -> None:

        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv3D(
            nc, weights, strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv1"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc, weights, strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="conv2"
        )
        self.use_vq = use_vq
        if use_vq:
            self.vq = VQBlock(
                vq_config["embeddings"], nc,
                vq_config["vq_beta"],
                name=f"{name}_vq"
            )

        # Normalisation
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:

        # 1st convolution
        x = self.conv1(x, training)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)

        # 2nd convolution
        x = self.conv2(x, training)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x)


#-------------------------------------------------------------------------
""" Up-sampling convolutional block """

class UpBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        nc: int,
        weights: tuple[int],
        strides: tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: dict,
        name: str | None = None
    ) -> None:

        super().__init__(name=name)
        self.tconv = tf.keras.layers.Conv3DTranspose(
            nc, weights, strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="tconv"
        )
        self.conv1 = tf.keras.layers.Conv3D(
            nc, weights, strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv1"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            nc, weights, strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv2"
        )

        self.use_vq = use_vq
        if use_vq:
            self.vq = VQBlock(
                vq_config["vq_embeddings"],
                nc, vq_config["vq_beta"],
                name=f"{name}_vq"
            )

        # Instance normalisation
        self.inst_norm_t = InstanceNorm(name="instancenormt")
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")

        self.concat = tf.keras.layers.Concatenate(name="concat")
    
    def call(
        self,
        x: tf.Tensor,
        skip: tf.Tensor,
        training: bool
    ) -> tf.Tensor:

        # Transpose convolution and up-sample
        x = self.tconv(x, training)
        x = self.inst_norm_t(x, training)
        x = tf.nn.relu(x)

        # 1st and 2nd convolutions
        x = self.concat([x, skip])
        x = self.conv1(x, training)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)
        x = self.conv2(x, training)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x)


#-------------------------------------------------------------------------
""" Up-sampling convolutional block without skip layer"""

class UpBlockNoSkip(tf.keras.layers.Layer):

    def __init__(
        self,
        nc: int,
        weights: tuple[int],
        strides: tuple[int],
        initialiser: tf.keras.initializers.Initializer,
        use_vq: bool,
        vq_config: dict,
        name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self.tconv = tf.keras.layers.Conv3DTranspose(
            nc, weights, strides=strides,
            padding="same",
            kernel_initializer=initialiser,
            name="tconv"
        )
        self.conv = tf.keras.layers.Conv3D(
            nc, weights, strides=(1, 1, 1),
            padding="same",
            kernel_initializer=initialiser,
            name="conv"
        )

        self.use_vq = use_vq
        if use_vq:
            self.vq = VQBlock(
                vq_config["vq_embeddings"],
                nc, vq_config["vq_beta"],
                name=f"{name}_vq"
            )

        # Instance normalisation
        self.inst_norm_1 = InstanceNorm(name="instancenorm1")
        self.inst_norm_2 = InstanceNorm(name="instancenorm2")
    
    def call(self, x: tf.Tensor, training: bool):

        x = self.tconv(x, training)
        x = self.inst_norm_1(x, training)
        x = tf.nn.relu(x)
        x = self.conv(x, training)
        x = self.inst_norm_2(x, training)

        # Perform vector quantization if necessary
        if self.use_vq:
            x = self.vq(x)

        return tf.nn.relu(x)


#-------------------------------------------------------------------------
""" Vector quantization layer -
    adapted from https://keras.io/examples/generative/vq_vae
    https://arxiv.org/pdf/2207.06189.pdf """

class VQBlock(tf.keras.layers.Layer):

    """ Input:
        - num_embeddings: number of possible embeddings
        - embedding_dim: size of features
        - beta: hyper-parameter for commitment loss """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, name=None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.initialiser = tf.keras.initializers.RandomUniform()
        self.dictionary = self.add_weight(
            "VQdict", shape=[embedding_dim, num_embeddings],
            initializer=self.initialiser, trainable=True
        )

    def call(self, x, t=None):
        img_dims = tf.shape(x)

        # Flatten img batch into matrix
        flat = tf.reshape(x, [-1, self.embedding_dim]) # NHWD X C

        # Quantization
        code_idx = self.get_code_indices(flat) # NHWD X 1
        idx_one_hot = tf.one_hot(code_idx, self.num_embeddings) # NHWD X K

        if self.embedding_dim == 1:
            quantized = tf.reduce_sum(idx_one_hot * self.dictionary, axis=1)
            quantized *= WEIGHT_SCALE
        else:
            quantized = tf.matmul(idx_one_hot, self.dictionary, transpose_b=True) # NHWD X C

        # Reshape back to normal dims
        q = tf.reshape(quantized, img_dims)

        # Get losses
        dictionary_loss = tf.reduce_mean(tf.square(tf.stop_gradient(x) - q))
        commitment_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(q)))
        self.add_loss(dictionary_loss + self.beta * commitment_loss)

        # Straight-through estimator
        q = x + tf.stop_gradient(q - x)

        return q

    def get_code_indices(self, flattened):
        similarity = tf.matmul(flattened, self.dictionary)
        distances = (
            tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.dictionary ** 2, axis=0, keepdims=True)
            - 2 * similarity
        ) # NHWD * K

        encoding_indices = tf.argmin(distances, axis=1) # NHWD X 1
        return encoding_indices
