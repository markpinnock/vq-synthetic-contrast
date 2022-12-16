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
        ([2, 2, 2], [4, 32, 32, 6, 4]),
        ([2, 2, 1], [4, 32, 32, 12, 4])
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

    in_dims = [4, 64, 64, 12, 4]
    img = tf.zeros(in_dims)
    x, skip = down(img)
    assert x.shape == out_dims
    assert skip.shape == in_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([2, 2, 2], [4, 64, 64, 12, 8]),
        ([2, 2, 1], [4, 64, 64, 6, 8])
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

    in_dims = [4, 32, 32, 6, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(out_dims)
    x = up(img, skip)
    assert x.shape == out_dims
