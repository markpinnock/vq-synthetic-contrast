import pytest
import tensorflow as tf

from vq_sce.networks.components.layers.conv_layers import (
    DownBlock,
    UpBlock
)


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([2, 2, 2], [4, 6, 32, 32, 4]),
        ([1, 2, 2], [4, 12, 32, 32, 4])
    ]
)
def test_DownBlock(strides, out_dims):
    init = tf.keras.initializers.HeNormal()

    down = DownBlock(
        nc=4,
        weights=[4, 4, 4],
        strides=strides,
        initialiser=init,
        use_vq=False,
        vq_config=None,
        name=None
    )

    in_dims = [4, 12, 64, 64, 4]
    img = tf.zeros(in_dims)
    x, skip = down(img)
    assert x.shape == out_dims
    assert skip.shape == in_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([2, 2, 2], [4, 12, 64, 64, 8]),
        ([1, 2, 2], [4, 6, 64, 64, 8])
    ]
)
def test_UpBlock(strides, out_dims):
    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=8,
        weights=[4, 4, 4],
        strides=strides,
        initialiser=init,
        use_vq=False,
        vq_config=None,
        name=None
    )

    in_dims = [4, 6, 32, 32, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(out_dims)
    x = up(img, skip)
    assert x.shape == out_dims
