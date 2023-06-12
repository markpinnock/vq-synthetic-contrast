from pathlib import Path
from typing import Any

import tensorflow as tf

from .darts_model import DARTSJointModel
from .model import JointModel, Model
from .multiscale_model import JointMultiscaleModel, MultiscaleModel

# -------------------------------------------------------------------------


def build_model_train(
    config: dict[str, Any],
    strategy: tf.distribute.Strategy,
    dev: bool = False,
) -> tf.keras.Model:
    """Build model for training purposes.
    :param config: config yaml
    :param dev: development mode
    :return: Keras model
    """
    scales = config["hyperparameters"]["scales"]
    expt_type = config["expt"]["expt_type"]
    optimisation_type = config["expt"]["optimisation_type"]

    with strategy.scope():
        if optimisation_type == "DARTS":
            assert len(scales) == 1 and expt_type == "joint", (scales, expt_type)

            model = DARTSJointModel(config)
            model.compile(  # type: ignore
                config["hyperparameters"]["opt"],
                config["hyperparameters"]["alpha_opt"],
                run_eagerly=dev,
            )

        else:
            if len(scales) == 1 and expt_type == "single":
                model = Model(config)
            elif len(scales) > 1 and expt_type == "single":
                model = MultiscaleModel(config)
            elif len(scales) == 1 and expt_type == "joint":
                model = JointModel(config)
            elif len(scales) > 1 and expt_type == "joint":
                model = JointMultiscaleModel(config)
            else:
                raise ValueError(scales, expt_type)

            model.compile(config["hyperparameters"]["opt"], run_eagerly=dev)
            model.build_model()

    return model


# -------------------------------------------------------------------------


def build_model_inference(
    expt_path: Path | list[Path],
    epoch: int | None = None,
) -> tf.keras.Model:
    """Load model weights for inference purposes.
    :param expt_path: path to the top level experiment folder
    :return: Keras model
    """
    # If models to be used sequentially in pipeline
    if isinstance(expt_path, list):
        assert len(expt_path) == 2, len(expt_path)
        models = []

        for path in expt_path:
            if epoch is None:
                ckpts = sorted(
                    list((path / "models").glob("*")),
                    key=lambda x: int(str(x.stem).split("-")[-1]),
                )
                assert len(ckpts) == 1, ckpts
                ckpt_path = ckpts[-1]

            else:
                ckpts = list(path / "models" / f"ckpt-{epoch}")
                assert len(ckpts) == 1, ckpts
                ckpt_path = ckpts[0]

            models.append(tf.keras.models.load_model(ckpt_path))

        model = tf.keras.Sequential(models)

    # Otherwise, load single model
    else:
        if epoch is None:
            ckpts = sorted(
                list((expt_path / "models").glob("*")),
                key=lambda x: int(str(x.stem).split("-")[-1]),
            )
            assert len(ckpts) == 1, ckpts
            ckpt_path = ckpts[-1]

        else:
            ckpts = list(expt_path / "models" / f"ckpt-{epoch}")
            assert len(ckpts) == 1, ckpts
            ckpt_path = ckpts[0]

        model = tf.keras.models.load_model(ckpt_path)

    return model
