import numpy as np
import tensorflow as tf

SCALE_FACTOR = 2


#-------------------------------------------------------------------------

def extract_patches(
    img: tf.Tensor,
    linear_coords: list[tf.Tensor],
    patch_size: tuple[int]
) -> tf.Tensor:

    linear_img = tf.reshape(img, -1)
    patches = []
    for coords in linear_coords:
        patch = tf.reshape(tf.gather(linear_img, coords), patch_size + [1])
        patches.append(patch)          

    patches = tf.stack(patches, axis=0)

    return patches


#-------------------------------------------------------------------------

def generate_indices(
    img_shape: tuple[int],
    strides: tuple[int],
    patch_size: tuple[int]
) -> list[tf.Tensor]:

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
