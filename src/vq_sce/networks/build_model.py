from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from .darts_model import DARTSJointModel, DARTSModel
from .model import JointModel, Model
from .multiscale_model import JointMultiscaleModel, MultiscaleModel

# -------------------------------------------------------------------------


def get_model(
    config: dict[str, Any],
    dev: bool = False,
    compile_model: bool = True,
) -> tf.keras.Model:
    scales = config["hyperparameters"]["scales"]
    expt_type = config["expt"]["expt_type"]
    optimisation_type = config["expt"]["optimisation_type"]

    if "darts" in optimisation_type:
        assert len(scales) == 1, scales

        if expt_type == "single":
            model = DARTSModel(config)
        else:
            model = DARTSJointModel(config)

        if compile_model:
            model.compile(
                config["hyperparameters"]["opt"],
                config["hyperparameters"]["darts_opt"],
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

        if compile_model:
            model.compile(  # type: ignore[call-arg]
                config["hyperparameters"]["opt"],
                run_eagerly=dev,
            )

    return model


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
    with strategy.scope():
        model = get_model(config, dev, compile_model=True)
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
            with open(path / "config.yml") as fp:
                config = yaml.load(fp, yaml.FullLoader)
            config["expt"]["local_mb_size"] = None
            model = get_model(config, compile_model=False)

            if epoch is None:
                ckpts = sorted(
                    list((path / "models").glob("*")),
                    key=lambda x: int(str(x.stem).split("-")[-1]),
                )
                ckpt_path = ckpts[-1]

            else:
                ckpts = list((path / "models" / f"ckpt-{epoch}").glob("*"))
                assert len(ckpts) == 1, ckpts
                ckpt_path = ckpts[0]

            model.load_weights(ckpt_path / "variables" / "variables")
            models.append(tf.keras.models.load_model(ckpt_path))

        model = tf.keras.Sequential(models)

    # Otherwise, load single model
    else:
        with open(expt_path / "config.yml") as fp:
            config = yaml.load(fp, yaml.FullLoader)
        config["expt"]["local_mb_size"] = None
        model = get_model(config, compile_model=False)

        if epoch is None:
            ckpts = sorted(
                list((expt_path / "models").glob(f"ckpt-{epoch}")),
                key=lambda x: int(str(x.stem).split("-")[-1]),
            )
            ckpt_path = ckpts[-1]

        else:
            ckpts = list((expt_path / "models").glob(f"ckpt-{epoch}"))
            assert len(ckpts) == 1, ckpts
            ckpt_path = ckpts[0]

        model.load_weights(ckpt_path / "variables" / "variables")

    return model
