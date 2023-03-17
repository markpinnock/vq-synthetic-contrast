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
    img_shape: tuple[int],
    strides: list[int],
    patch_size: list[int]
) -> list[tf.Tensor]:
    """Generate indices of flattened image patches to extract.
    :param img_shape: size of image to be processes
    :param strides: strides of patches
    :param patch_size: size of patches to be extracted e.g. [D, H, W]
    Returns: list of flattened patch indices
    """
    # Linear coords are what we'll use to do our patch updates in 1D
    # E.g. [1, 2, 3
    #       4, 5, 6
    #       7, 8, 9]

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
