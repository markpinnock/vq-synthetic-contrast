import pytest
import tensorflow as tf

from vq_sce.networks.components.layers.conv_layers import (
    BottomBlock,
    DownBlock,
    UpBlock
)
from vq_sce.networks.components.layers.vq_layers import VQBlock


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([4, 2, 2], [2, 3, 32, 32, 4]),
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

    in_dims = [out_dims[0], 12, 64, 64, 4]
    img = tf.zeros(in_dims)
    x, skip = down(img)
    assert x.shape == out_dims
    assert skip.shape == in_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([2, 2, 2], [4, 3, 16, 16, 4]),
        ([1, 2, 2], [4, 12, 16, 16, 4])
    ]
)
def test_BottomBlock(strides: list[int], out_dims: list[int]) -> None:
    init = tf.keras.initializers.HeNormal()

    bottom = BottomBlock(
        nc=4,
        weights=[4, 4, 4],
        strides=strides,
        initialiser=init,
        use_vq=False,
        vq_config=None,
        shared_vq=None,
        name=None
    )

    in_dims = [out_dims[0], 12, 64, 64, 4]
    img = tf.zeros(in_dims)
    x = bottom(img)
    assert x.shape == out_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims",
    [
        ([4, 2, 2], [2, 24, 64, 64, 4]),
        ([2, 2, 2], [4, 12, 64, 64, 8]),
        ([1, 2, 2], [4, 6, 64, 64, 8])
    ]
)
def test_UpBlock(strides: list[int], out_dims: list[int]) -> None:
    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=out_dims[-1],
        weights=[4, 4, 4],
        strides=strides,
        upsamp_factor=1,
        initialiser=init,
        use_vq=False,
        vq_config=None,
        name=None
    )

    in_dims = [out_dims[0], 6, 32, 32, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(out_dims)
    x = up(img, skip)
    assert x.shape == out_dims


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strides,out_dims,skip_dims",
    [
        ([1, 2, 2], [4, 6, 64, 64, 8], [4, 3, 64, 64, 8]),
        ([2, 2, 2], [4, 12, 64, 64, 8], [4, 3, 64, 64, 8]),
        ([4, 2, 2], [4, 24, 64, 64, 8], [4, 3, 64, 64, 8]),
        ([4, 2, 2], [2, 24, 64, 64, 4], [2, 6, 64, 64, 4]),
    ]
)
def test_UpsampleSkip(
    strides: list[int],
    out_dims: list[int],
    skip_dims: list[int]
) -> None:

    upsamp_factor = out_dims[1] // skip_dims[1]
    init = tf.keras.initializers.HeNormal()

    up = UpBlock(
        nc=out_dims[-1],
        weights=[4, 4, 4],
        strides=strides,
        upsamp_factor=upsamp_factor,
        initialiser=init,
        use_vq=False,
        vq_config=None,
        name=None
    )

    in_dims = [out_dims[0], 6, 32, 32, 4]
    img = tf.zeros(in_dims)
    skip = tf.zeros(skip_dims)
    x = up(img, skip)
    assert x.shape == out_dims
