import numpy as np
import tensorflow as tf

SCALE_FACTOR = 2


#-------------------------------------------------------------------------

def scale_indices(coords: tf.Tensor):
    img = tf.zeros((4, 4, 4))
    H, W, D = img.shape
    idx_map = tf.reshape(tf.range(tf.reduce_prod((D, H, W))), (D, H, W))
    coords = generate_indices(img, 2, [2, 2, 2])[1]
    D_offset = coords[0, 0, 1] - coords[0, 0, 0]
    H_offset = coords[1, 0, 0] - coords[0, 0, 0]
    W_offset = coords[0, 1, 0] - coords[0, 0, 0]
    first_element = coords[0, 0, 0] * SCALE_FACTOR
    zs = (tf.constant(coords[0, 0, :]) * H_offset * 4)[:, tf.newaxis, tf.newaxis]
    ys = (first_element + tf.range(W_offset))
    xs = (first_element + tf.range(W_offset) * W_offset * 2)[:, tf.newaxis]
    print(coords)
    print(idx_map)
    new_coords = zs + ys + xs
    new_idx_map = tf.reshape(tf.range(tf.reduce_prod((D, H * 2, W * 2))), (D, H * 2, W * 2))
    print(new_coords)
    print(new_idx_map)
    exit()


#-------------------------------------------------------------------------

def extract_patches(
    img: tf.Tensor,
    linear_coords: list[tf.Tensor],
    patch_size: list[int] | tuple[int]
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
    img: tf.Tensor,
    stride_length: int,
    patch_size: list[int] | tuple[int]
) -> list[tf.Tensor]:

    # Linear coords are what we'll use to do our patch updates in 1D
    # E.g. [1, 2, 3
    #       4, 5, 6
    #       7, 8, 9]

    D, H, W = img.shape
    idx_map = tf.reshape(tf.range(tf.reduce_prod(img.shape)), img.shape)
    indices = []

    for k in range(0, D - patch_size[0] + stride_length, stride_length):
        if (k + patch_size[0] > D):  
            for j in range(0, H - patch_size[1] + stride_length, stride_length):
                for i in range(0, W - patch_size[2] + stride_length, stride_length):
                    linear_indices = idx_map[-patch_size[0]:, j:(j + patch_size[1]), i:(i + patch_size[2])]
                    indices.append(linear_indices)

            return indices

        for j in range(0, H - patch_size[1] + stride_length, stride_length):
            for i in range(0, W - patch_size[2] + stride_length, stride_length):
                linear_indices = idx_map[k:(k + patch_size[0]), j:(j + patch_size[1]), i:(i + patch_size[2])]
                indices.append(linear_indices)

    return indices
