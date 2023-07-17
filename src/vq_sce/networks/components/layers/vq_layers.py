import numpy as np
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
        task_lr: float = 1.0,
        beta: float = 0.25,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, dtype="float32")
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

        # Learning rate for this task
        self.task_lr = task_lr

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

        # Multiply by task learning rate
        quantized = self.task_lr * quantized

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


# -------------------------------------------------------------------------


class DARTSVQBlock(tf.keras.layers.Layer):
    """Vector quantization layer.
    Adapted from https://keras.io/examples/generative/vq_vae
    https://arxiv.org/pdf/2207.06189.pdf

    :param num_embeddings: number of possible embeddings
    :param embedding_dim: size of features
    :param beta: hyper-parameter for commitment loss
    """

    # Gamma: weights for candidate dictionaries in this block
    alpha_vq: tf.Variable

    def __init__(
        self,
        num_embeddings: list[int],
        embedding_dim: int,
        task_lr: float = 1.0,
        beta: float = 0.25,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, dtype="float32")
        self.embedding_dim = embedding_dim
        self.beta = beta

        if embedding_dim == 1:
            raise ValueError("Embedding dim = 1 not supported in DARTS")

        self.initialiser = tf.keras.initializers.RandomUniform()
        num_embeddings_log2 = np.log2(num_embeddings)
        self.dictionaries = []

        for dim_log2 in range(
            int(num_embeddings_log2[0]),
            int(num_embeddings_log2[-1] + 1),
        ):
            self.dictionaries.append(
                self.add_weight(
                    f"VQdict_{dim_log2}",
                    shape=[embedding_dim, np.power(2, dim_log2)],
                    initializer=self.initialiser,
                    trainable=True,
                ),
            )

        # Learning rate for this task
        self.task_lr = task_lr

        self.softmax = tf.keras.layers.Activation("softmax", dtype="float32")
        self.num_dictionaries = len(self.dictionaries)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        img_dims = tf.shape(x)
        alpha_vq = tf.squeeze(self.softmax(self.alpha_vq))

        # Flatten img batch into matrix
        flat = tf.reshape(x, [-1, self.embedding_dim])  # NHWD X C

        dictionary_loss = []
        commitment_loss = []
        q = []

        for i in range(self.num_dictionaries):
            # Quantization
            num_embeddings = self.dictionaries[i].shape[1]
            code_idx = self.get_code_indices(flat, self.dictionaries[i])  # NHWD X 1
            idx_one_hot = tf.one_hot(code_idx, num_embeddings)  # NHWD X K

            quantized = tf.matmul(
                idx_one_hot,
                self.dictionaries[i],
                transpose_b=True,
            )  # NHWD X C

            q.append(quantized)

            # Get losses
            dictionary_loss.append(
                tf.reduce_mean(
                    tf.square(tf.stop_gradient(flat) - quantized),
                ),
            )
            commitment_loss.append(
                tf.reduce_mean(
                    tf.square(flat - tf.stop_gradient(quantized)),
                ),
            )

        dictionary_loss_tensor = tf.stack(dictionary_loss, axis=0)
        commitment_loss_tensor = tf.stack(commitment_loss, axis=0)

        weighted_dictionary_loss = tf.reduce_sum(dictionary_loss_tensor * alpha_vq)
        weighted_commitment_loss = tf.reduce_sum(commitment_loss_tensor * alpha_vq)

        self.add_loss(weighted_dictionary_loss + self.beta * weighted_commitment_loss)

        # Get weighted average and reshape back to normal dims
        weighted_q = tf.reduce_sum(
            tf.stack(q, axis=0) * alpha_vq[:, tf.newaxis, tf.newaxis],
            axis=0,
        )
        weighted_q = tf.reshape(weighted_q, img_dims)

        # Multiply by task learning rate
        weighted_q = self.task_lr * weighted_q

        # Straight-through estimator
        weighted_q = x + tf.stop_gradient(weighted_q - x)

        return weighted_q

    def get_code_indices(
        self,
        flattened: tf.Tensor,
        dictionary: tf.Tensor,
    ) -> tf.Tensor:
        similarity = tf.matmul(flattened, dictionary)
        distances = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            + tf.reduce_sum(dictionary**2, axis=0, keepdims=True)
            - 2 * similarity
        )  # NHWD * K

        encoding_indices = tf.argmin(distances, axis=1)  # NHWD X 1

        return encoding_indices
