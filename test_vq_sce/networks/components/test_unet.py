import numpy as np
import pytest
import tensorflow as tf

from vq_sce.networks.components.unet import MultiscaleUNet, UNet

CONFIG = {
    "source_dims": [4, 16, 16],
    "target_dims": [4, 16, 16],
    "nc": 4,
    "layers": 2,
    "upsample_layer": False,
    "residual": False,
    "vq_layers": None,
    "vq_beta:": None,
}


# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth,img_dims",
    [
        (2, [1, 16, 16, 16, 1]),
        (3, [2, 8, 16, 16, 1]),
        (4, [2, 4, 16, 16, 1]),
        (3, [4, 16, 32, 32, 1]),
        (4, [4, 8, 32, 32, 1]),
        (5, [4, 4, 32, 32, 1]),
    ],
)
def test_unet_output(depth: int, img_dims: list[int]) -> None:
    """Test UNet output is correct size"""

    config = dict(CONFIG)
    config["layers"] = depth
    config["source_dims"] = img_dims[1:-1]
    config["target_dims"] = img_dims[1:-1]
    init = tf.keras.initializers.Zeros()

    model = UNet(init, config)
    img = tf.zeros(img_dims)
    out = model(img)

    assert img.shape == out.shape


# -------------------------------------------------------------------------


def test_multiscale_unet() -> None:
    """Test UNet up-sampling layer"""

    config = dict(CONFIG)
    init = tf.keras.initializers.Zeros()
    dims = config["source_dims"]

    model = MultiscaleUNet(init, config)
    img = tf.ones([2] + dims + [1])
    exp_out_dims = [2] + [dims[0], dims[1] * 2, dims[2] * 2] + [1]
    out, _ = model(img)

    assert out.shape == exp_out_dims


# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth,in_dims,out_dims",
    [
        (2, [1, 3, 16, 16, 1], [1, 12, 16, 16, 1]),
        (3, [2, 8, 16, 16, 1], [2, 32, 16, 16, 1]),
        (4, [2, 4, 16, 16, 1], [2, 8, 16, 16, 1]),
        (3, [4, 16, 32, 32, 1], [4, 32, 32, 32, 1]),
        (4, [4, 5, 32, 32, 1], [4, 20, 32, 32, 1]),
        (5, [4, 6, 32, 32, 1], [4, 48, 32, 32, 1]),
    ],
)
def test_asymmetric_unet_output(
    depth: int, in_dims: list[int], out_dims: list[int]
) -> None:
    """Test UNet output is correct size with different
    input and output depths
    """

    config = dict(CONFIG)
    config["layers"] = depth
    config["source_dims"] = in_dims[1:-1]
    config["target_dims"] = out_dims[1:-1]
    init = tf.keras.initializers.Zeros()

    model = UNet(init, config)
    img = tf.zeros(in_dims)
    out = model(img)

    assert out.shape == out_dims
