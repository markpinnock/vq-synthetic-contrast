import numpy as np
import pytest
import tensorflow as tf

from vq_sce.networks.components.unet import UNet

CONFIG = {
    "img_dims": [4, 16, 16],
    "nc": 4,
    "layers": 2,
    "upsample_layer": False,
    "residual": False,
    "vq_layers": None,
    "vq_beta:": None
}


#-------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth,img_dims",
    [
        (2, [1, 16, 16, 16, 1]),
        (3, [2, 8, 16, 16, 1]),
        (4, [2, 4, 16, 16, 1]),
        (3, [4, 16, 32, 32, 1]),
        (4, [4, 8, 32, 32, 1]),
        (5, [4, 4, 32, 32, 1])
    ]
)
def test_unet_output(depth: int, img_dims: list[int]) -> None:
    """ Test UNet output is correct size """

    config = dict(CONFIG)
    config["layers"] = depth
    config["img_dims"] = img_dims[1:-1]
    init = tf.keras.initializers.Zeros()

    model = UNet(init, config)
    img = tf.zeros(img_dims)
    out, _ = model(img)

    assert img.shape == out.shape


#-------------------------------------------------------------------------

def test_residual_unet() -> None:
    """ Test residual UNet """

    config = dict(CONFIG)
    config["residual"] = True
    init = tf.keras.initializers.Zeros()

    model = UNet(init, config)
    img = tf.ones([2] + config["img_dims"] + [1])
    out, _ = model(img)

    assert np.equal(img.numpy(), out.numpy()).all()


#-------------------------------------------------------------------------

@pytest.mark.parametrize("residual", [False, True])
def test_upsample_unet(residual: bool) -> None:
    """ Test UNet up-sampling layer """

    config = dict(CONFIG)
    config["residual"] = residual
    config["upsample_layer"] = True
    init = tf.keras.initializers.Zeros()
    dims = config["img_dims"]

    model = UNet(init, config)
    img = tf.ones([2] + dims + [1])
    exp_out_dims = [2] + [dims[0], dims[1] * 2, dims[2] * 2] + [1]
    out, _ = model(img)

    assert out.shape == exp_out_dims

    if residual:
        res = tf.ones(exp_out_dims)
        assert np.equal(res.numpy(), out.numpy()).all()
