from typing import Any

import tensorflow as tf
import yaml

from .darts_model import DARTSJointModel
from .model import DualModel, JointModel, Model
from .multiscale_model import JointMultiscaleModel, MultiscaleModel

MODEL_DICT = {
    "single_scale": Model,
    "multi_scale": MultiscaleModel,
    "single_joint": JointModel,
    "multi_joint": JointMultiscaleModel,
    "single_dual": DualModel,
}


def build_model(
    config: dict[str, Any],
    purpose: str = "training",
    dev: bool = False,
) -> tf.keras.Model:
    scales = config["hyperparameters"]["scales"]
    expt_type = config["expt"]["expt_type"]

    try:
        optimisation_type = config["expt"]["optimisation_type"]
    except KeyError:
        optimisation_type = None

    if optimisation_type != "DARTS":
        if len(scales) == 1 and expt_type == "single":
            model = Model(config)
        elif len(scales) > 1 and expt_type == "single":
            model = MultiscaleModel(config)
        elif len(scales) == 1 and expt_type == "joint":
            model = JointModel(config)
        elif len(scales) > 1 and expt_type == "joint":
            model = JointMultiscaleModel(config)
        elif len(scales) == 1 and expt_type == "dual":
            with open(config["paths"]["ce_path"] / "config.yml") as fp:
                ce_config = yaml.load(fp, yaml.FullLoader)
            with open(config["paths"]["sr_path"] / "config.yml") as fp:
                sr_config = yaml.load(fp, yaml.FullLoader)

            model = DualModel(sr_config, ce_config)
            assert purpose == "inference"
        else:
            raise ValueError(scales, expt_type)

    else:
        if len(scales) == 1 and expt_type == "joint":
            model = DARTSJointModel(config)
        else:
            raise ValueError(scales, expt_type)

    if purpose == "training":
        if optimisation_type != "DARTS":
            model.compile(config["hyperparameters"]["opt"], run_eagerly=dev)
        else:
            model.compile(
                config["hyperparameters"]["opt"],
                config["hyperparameters"]["alpha_opt"],
                run_eagerly=dev,
            )
        model.build_model()
        return model
    elif purpose == "inference" and expt_type == "single":
        model.build_model()
        model.UNet.load_weights(config["paths"]["expt_path"] / "models" / "model.ckpt")
        return model
    elif purpose == "inference" and expt_type == "joint":
        model.build_model()
        model.ce_UNet.load_weights(
            config["paths"]["expt_path"] / "models" / "ce_model.ckpt",
        )
        model.sr_UNet.load_weights(
            config["paths"]["expt_path"] / "models" / "sr_model.ckpt",
        )
        return model
    elif expt_type == "dual":
        model.build_model()
        model.ce_UNet.load_weights(config["paths"]["ce_path"] / "models" / "model.ckpt")
        model.sr_UNet.load_weights(config["paths"]["sr_path"] / "models" / "model.ckpt")
        return model
    else:
        raise ValueError("Purpose must be 'training' or 'inference'")
