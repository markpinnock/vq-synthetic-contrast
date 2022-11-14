import numpy as np
import tensorflow as tf


def extract_patches(
    img: tf.Tensor,
    stride_length:
    int, patch_size: int
) -> tuple[tf.Tensor, list[int]]:

    H, W, D = img.shape
    patches = []
    indices = []

    for k in range(0, D - patch_size[2] + stride_length, stride_length):
        if (k + patch_size[2]) > D:  
            for j in range(0, W - patch_size[1] + stride_length, stride_length):
                for i in range(0, H - patch_size[0] + stride_length, stride_length):
                    patches.append(img[i:(i + patch_size[0]), j:(j + patch_size[1]), -patch_size[2]:, np.newaxis])
                    indices.append([i, j, -patch_size[2]])

            return patches, indices

        for j in range(0, W - patch_size[1] + stride_length, stride_length):
            for i in range(0, H - patch_size[0] + stride_length, stride_length):
                patches.append(img[i:(i + patch_size[0]), j:(j + patch_size[1]), k:(k + patch_size[2]), np.newaxis])
                indices.append([i, j, k])

        return patches, indices


def generate_indices(
    img: tf.Tensor,
    stride_length: int,
    patch_size: int
) -> list[int]:

    # Linear coords are what we'll use to do our patch updates in 1D
    # E.g. [1, 2, 3
    #       4, 5, 6
    #       7, 8, 9]

    H, W, D = img.shape
    idx_map = tf.reshape(tf.range(tf.reduce_prod(img.shape)), img.shape)
    indices = []

    for i in range(0, H - patch_size[0] + stride_length, stride_length):
        for j in range(0, W - patch_size[1] + stride_length, stride_length):
            for k in range(0, D - patch_size[2] + stride_length, stride_length):

                if (k + patch_size[2]) > D:
                    linear_indices = idx_map[i:(i + patch_size[0]), j:(j + patch_size[1]), -patch_size[2]:]
                    indices.append(linear_indices)

                else:
                    linear_indices = idx_map[i:(i + patch_size[0]), j:(j + patch_size[1]), k:(k + patch_size[2])]
                    indices.append(linear_indices)

    return indices
