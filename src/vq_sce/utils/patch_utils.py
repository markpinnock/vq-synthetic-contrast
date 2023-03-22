import numpy as np
import tensorflow as tf

SCALE_FACTOR = 2


#-------------------------------------------------------------------------

def extract_patches(
    img: tf.Tensor,
    linear_indices: list[tf.Tensor],
    patch_size: list[int]
) -> tf.Tensor:
    """Extract patches from images.
    :param linear_indices: list of pixel indices of flattened image to extract
    :param patch_size: size of patches to be extracted e.g. [D, H, W]
    Returns: N patches in tensor of size [N, D, H, W]
    """
    linear_img = tf.reshape(img, -1)
    patches = []
    for idx in linear_indices:
        patch = tf.reshape(tf.gather(linear_img, idx), patch_size + [1])
        patches.append(patch)          

    patches = tf.stack(patches, axis=0)

    return patches


#-------------------------------------------------------------------------

def generate_indices(
    img_shape: tuple[int, int, int],
    strides: list[int],
    patch_size: list[int]
) -> list[tf.Tensor]:
    """Generate indices of flattened image patches to extract.
    :param img_shape: size of image to be processes
    :param strides: strides of patches
    :param patch_size: size of patches to be extracted e.g. [D, H, W]
    Returns: list of flattened patch indices

    Linear coords are what we'll use to do our patch updates in 1D
    E.g. [1, 2, 3
          4, 5, 6
          7, 8, 9]

    """
    D, H, W = img_shape
    idx_map = tf.reshape(tf.range(tf.reduce_prod(img_shape)), img_shape)
    indices = []

    for k in range(0, D - patch_size[0] + strides[0], strides[0]):
        if (k + patch_size[0] > D):  
            for j in range(0, H - patch_size[1] + strides[1], strides[1]):
                for i in range(0, W - patch_size[2] + strides[2], strides[2]):
                    linear_indices = idx_map[-patch_size[0]:, j:(j + patch_size[1]), i:(i + patch_size[2])]
                    indices.append(linear_indices)

            return indices

        for j in range(0, H - patch_size[1] + strides[1], strides[1]):
            for i in range(0, W - patch_size[2] + strides[2], strides[2]):
                linear_indices = idx_map[k:(k + patch_size[0]), j:(j + patch_size[1]), i:(i + patch_size[2])]
                indices.append(linear_indices)

    return indices


#-------------------------------------------------------------------------

class CombinePatches:

    """Recombine extracted image patches into original image."""

    linear_weights: tf.Tensor
    linear: tf.Tensor

    def new_subject(self, subject_dims: tuple[int, int, int]) -> None:
        """Pass dims of image for new subject."""
        self.DHW_dims = subject_dims
        self.linear_img_size = tf.reduce_prod(self.DHW_dims)
        self._reset()

    def get_img(self) -> tf.Tensor:
        """Return reconstructed image."""
        linear = self.linear / self.linear_weights
        img = tf.reshape(linear, self.DHW_dims)

        return img

    def _reset(self) -> None:
        self.linear = tf.zeros(self.linear_img_size)
        self.linear_weights = np.zeros(self.linear_img_size)

    def apply_patches(self, patches: tf.Tensor, indices: tf.Tensor) -> None:
        """Pass patches for reconstruction.
        :param patches: tensor of patches
        :param indices: tensor of flattened patch indices
        """
        # Flatten minibatch of indices
        indices = tf.reshape(indices, [-1, 1])
        update = tf.reshape(patches, -1)

        # Update 1D image with patches
        self.linear = tf.tensor_scatter_nd_add(self.linear, indices, update)

        # Update weights
        self.linear_weights = tf.tensor_scatter_nd_add(self.linear_weights, indices, tf.ones_like(update))
