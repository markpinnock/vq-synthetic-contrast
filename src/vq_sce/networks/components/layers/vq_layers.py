import tensorflow as tf

WEIGHT_SCALE = 0.02


# -------------------------------------------------------------------------


class VQBlock(tf.keras.layers.Layer):
    """Vector quantization layer.
    Adapted from https://keras.io/examples/generative/vq_vae
    https://arxiv.org/pdf/2207.06189.pdf

    :param num_embeddings: number of possible embeddings
    :param embedding_dim: size of features
    :param beta: hyper-parameter for commitment loss
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.initialiser = tf.keras.initializers.RandomUniform()
        self.dictionary = self.add_weight(
            "VQdict",
            shape=[embedding_dim, num_embeddings],
            initializer=self.initialiser,
            trainable=True,
        )

        # Alpha, learning rate for this block (set to constant 0.5 if not DARTS)
        self.vq_alpha = self.add_weight(
            "alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=False,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        img_dims = tf.shape(x)

        # Flatten img batch into matrix
        flat = tf.reshape(x, [-1, self.embedding_dim])  # NHWD X C

        # Quantization
        code_idx = self.get_code_indices(flat)  # NHWD X 1
        idx_one_hot = tf.one_hot(code_idx, self.num_embeddings)  # NHWD X K

        if self.embedding_dim == 1:
            quantized = tf.reduce_sum(idx_one_hot * self.dictionary, axis=1)
            quantized *= WEIGHT_SCALE
        else:
            quantized = tf.matmul(
                idx_one_hot,
                self.dictionary,
                transpose_b=True,
            )  # NHWD X C

        # Multiply by block learning rate
        quantized = self.vq_alpha * quantized

        # Reshape back to normal dims
        q = tf.reshape(quantized, img_dims)

        # Get losses
        dictionary_loss = tf.reduce_mean(tf.square(tf.stop_gradient(x) - q))
        commitment_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(q)))
        self.add_loss(dictionary_loss + self.beta * commitment_loss)

        # Straight-through estimator
        q = x + tf.stop_gradient(q - x)

        return q

    def get_code_indices(self, flattened: tf.Tensor) -> tf.Tensor:
        similarity = tf.matmul(flattened, self.dictionary)
        distances = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.dictionary**2, axis=0, keepdims=True)
            - 2 * similarity
        )  # NHWD * K

        encoding_indices = tf.argmin(distances, axis=1)  # NHWD X 1

        return encoding_indices
