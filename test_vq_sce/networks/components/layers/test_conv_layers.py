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
def test_DownBlock(strides: list[int], out_dims: list[int]) -> None:
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
def test_UpBlock(strides: list[int], out_dims: list[int]) -> None:
    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=8,
        weights=[4, 4, 4],
        strides=strides,
        upsamp_factor=1,
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


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,in_dims,out_dims",
    [
        ([2, 2, 2], [4, 3, 64, 64, 8], [4, 12, 64, 64, 8]),
        ([1, 2, 2], [4, 6, 64, 64, 8], [4, 12, 64, 64, 8])
    ]
)
def test_UpsampleSkip(
    strides: list[int],
    in_dims: list[int],
    out_dims: list[int]
) -> None:

    upsamp_factor = out_dims[1] // in_dims[1]
    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=8,
        weights=[4, 1, 1],
        strides=strides,
        upsamp_factor=upsamp_factor,
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
