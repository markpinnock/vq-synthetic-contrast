from pathlib import Path
from typing import Any

import pytest
import tensorflow as tf
import yaml

from vq_sce.networks.build_model import build_model_train

# -------------------------------------------------------------------------


@pytest.fixture
def config() -> dict[str, Any]:
    """Get model config."""
    config_path = Path(__file__).parent
    with open(config_path / "config.yml", "r") as fp:
        config = yaml.load(fp, yaml.FullLoader)
    return config


# -------------------------------------------------------------------------


def test_no_vq_block(config: dict[str, Any]):
    """Test that no VQ block added to network."""
    config["expt"]["expt_type"] = "single"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = None
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.UNet.vq_block is None


# -------------------------------------------------------------------------


def test_one_vq_block(config: dict[str, Any]):
    """Test size of one VQ block."""
    config["expt"]["expt_type"] = "single"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = {"bottom": 4}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.UNet.vq_block.num_dictionaries == 1
    assert model.UNet.vq_block.dictionaries[0].get_shape()[1] == 4


# -------------------------------------------------------------------------


def test_two_vq_blocks(config: dict[str, Any]):
    """Test size of multiple VQ blocks."""
    config["expt"]["expt_type"] = "single"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = {"bottom": [4, 16]}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.UNet.vq_block.num_dictionaries == 2
    assert model.UNet.vq_block.dictionaries[0].get_shape()[1] == 4
    assert model.UNet.vq_block.dictionaries[1].get_shape()[1] == 16


# -------------------------------------------------------------------------


def test_darts_vq_block(config: dict[str, Any]):
    """Test size of VQ blocks in DARTS formulation."""
    config["expt"]["expt_type"] = "single"
    config["expt"]["optimisation_type"] = "darts-vq"
    config["hyperparameters"]["vq_layers"] = {"bottom": [4, 16]}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.UNet.vq_block.num_dictionaries == 3
    assert model.UNet.vq_block.dictionaries[0].get_shape()[1] == 4
    assert model.UNet.vq_block.dictionaries[1].get_shape()[1] == 8
    assert model.UNet.vq_block.dictionaries[2].get_shape()[1] == 16


# -------------------------------------------------------------------------


def test_no_vq_block_joint(config: dict[str, Any]):
    """Test that no VQ block added to joint networks."""
    config["expt"]["expt_type"] = "joint"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = None
    strategy = tf.distribute.MirroredStrategy()

    with pytest.raises(TypeError):
        _ = build_model_train(config, strategy)


# -------------------------------------------------------------------------


def test_one_vq_block_joint_(config: dict[str, Any]):
    """Test size of one VQ block in joint networks."""
    config["expt"]["expt_type"] = "joint"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = {"bottom": 4}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.sr_UNet.vq_block.num_dictionaries == 1
    assert model.sr_UNet.vq_block.dictionaries[0].get_shape()[1] == 4

    assert (
        model.sr_UNet.vq_block.dictionaries[0] is model.ce_UNet.vq_block.dictionaries[0]
    )


# -------------------------------------------------------------------------


def test_two_vq_blocks_joint(config: dict[str, Any]):
    """Test size of multiple VQ blocks in joint networks."""
    config["expt"]["expt_type"] = "joint"
    config["expt"]["optimisation_type"] = "simple"
    config["hyperparameters"]["vq_layers"] = {"bottom": [4, 16]}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.sr_UNet.vq_block.num_dictionaries == 2
    assert model.sr_UNet.vq_block.dictionaries[0].get_shape()[1] == 4
    assert model.sr_UNet.vq_block.dictionaries[1].get_shape()[1] == 16

    assert (
        model.sr_UNet.vq_block.dictionaries[0] is model.ce_UNet.vq_block.dictionaries[0]
    )
    assert (
        model.sr_UNet.vq_block.dictionaries[1] is model.ce_UNet.vq_block.dictionaries[1]
    )


# -------------------------------------------------------------------------


def test_darts_vq_block_joint(config: dict[str, Any]):
    """Test size of VQ blocks in DARTS formulation for joint networks."""
    config["expt"]["expt_type"] = "joint"
    config["expt"]["optimisation_type"] = "darts-vq"
    config["hyperparameters"]["vq_layers"] = {"bottom": [4, 16]}
    strategy = tf.distribute.MirroredStrategy()

    model = build_model_train(config, strategy)
    assert model.sr_UNet.vq_block.num_dictionaries == 3
    assert model.sr_UNet.vq_block.dictionaries[0].get_shape()[1] == 4
    assert model.sr_UNet.vq_block.dictionaries[1].get_shape()[1] == 8
    assert model.sr_UNet.vq_block.dictionaries[2].get_shape()[1] == 16

    assert (
        model.sr_UNet.vq_block.dictionaries[0] is model.ce_UNet.vq_block.dictionaries[0]
    )
    assert (
        model.sr_UNet.vq_block.dictionaries[1] is model.ce_UNet.vq_block.dictionaries[1]
    )
    assert (
        model.sr_UNet.vq_block.dictionaries[2] is model.ce_UNet.vq_block.dictionaries[2]
    )
