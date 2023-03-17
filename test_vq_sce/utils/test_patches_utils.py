import numpy as np
import os
import pytest
import tensorflow as tf

from vq_sce.utils.patch_utils import extract_patches, generate_indices


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "img_size,patch_size,strides",
    [
        ((1, 4, 4), [1, 2, 2], [1, 1, 1]),
        ((1, 4, 4), [1, 2, 2], [1, 2, 2]),
        ((1, 32, 32), [1, 8, 8], [1, 4, 4]),
        ((1, 32, 32), [1, 16, 16], [1, 8, 8]),
    ]
)
def test_generate_indices_2d(
    img_size: tuple[int, int, int],
    patch_size: list[int],
    strides: list[int]
) -> None:
    """Test indices of patches to be extracted in 2D."""
    def _setup_ground_truth(
        img: tf.Tensor,
        sizes: list[int],
        strides: list[int]
    ) -> tf.Tensor:
        patch = tf.image.extract_patches(
            img[:, :, :, tf.newaxis],
            sizes=sizes + [1],
            strides=strides + [1],
            rates=(1, 1, 1, 1),
            padding="VALID"
        )

        patch = tf.reshape(patch, [
            tf.reduce_prod(patch.shape[1:3])]
         + sizes[1:3] + [1])

        return patch

    test_img = tf.reshape(tf.range(tf.reduce_prod(img_size)), img_size)
    ground_truth = _setup_ground_truth(test_img, patch_size, strides)
    pred_indices = tf.stack(generate_indices(img_size, strides, patch_size), axis=0)

    assert np.equal(
        ground_truth.numpy().squeeze(),
        pred_indices.numpy().squeeze()
    ).all()


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "img_size,patch_size,strides",
    [
        ((1, 4, 4), [1, 2, 2], [1, 1, 1]),
        ((1, 4, 4), [1, 2, 2], [1, 2, 2]),
        ((1, 32, 32), [1, 8, 8], [1, 4, 4]),
        ((1, 32, 32), [1, 16, 16], [1, 8, 8]),
        ((2, 64, 64), [2, 8, 8], [2, 4, 4]),
        ((4, 64, 64), [4, 16, 16], [2, 8, 8]),
    ]
)
def test_extract_patches(
    img_size: tuple[int, int, int],
    patch_size: list[int],
    strides: list[int]
) -> None:
    """Test 2D and 3D patch extraction."""
    test_img = tf.reshape(tf.range(tf.reduce_prod(img_size)), img_size)
    indices = generate_indices(test_img.shape, strides, patch_size)
    ground_truth = tf.stack(indices, axis=0)
    pred_patches = extract_patches(test_img, indices, patch_size)

    assert np.equal(
        ground_truth.numpy().squeeze(),
        pred_patches.numpy().squeeze()
    ).all()
