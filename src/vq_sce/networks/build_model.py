import tensorflow as tf
from typing import Any

from .model import Model, JointModel
from .multiscale_model import MultiscaleModel, JointMultiscaleModel

MODEL_DICT = {
    "single_scale": Model,
    "multi_scale": MultiscaleModel,
    "single_joint": JointModel,
    "multi_joint": JointMultiscaleModel
}


def build_model(config: dict[str, Any], purpose: str = "training") -> tf.keras.Model:
    scales = config["hyperparameters"]["scales"]
    expt_type = config["expt"]["expt_type"]
    if len(scales) == 1 and expt_type == "single":
        model_type = "single_scale"
    elif len(scales) > 1 and expt_type == "single":
        model_type = "multi_scale"
    elif len(scales) == 1 and expt_type == "joint":
        model_type = "single_joint"
    elif len(scales) > 1 and expt_type == "joint":
        model_type = "multi_joint"
    else:
        raise ValueError(scales, expt_type)

    model: tf.keras.Model = MODEL_DICT[model_type](config)

    if purpose == "training":
        optimiser = tf.keras.optimizers.Adam(*config["hyperparameters"]["opt"], name="opt")
        model.compile(optimiser)
        return model
    elif purpose == "inference" and expt_type == "single":
        model.build_model()
        model.UNet.load_weights(config["paths"]["expt_path"] / "models" / "model.ckpt")
        return model
    elif purpose == "inference" and expt_type == "joint":
        model.build_model()
        model.ce_UNet.load_weights(config["paths"]["expt_path"] / "models" / "ce_model.ckpt")
        model.sr_UNet.load_weights(config["paths"]["expt_path"] / "models" / "sr_model.ckpt")
        return model
    else:
        raise ValueError("Purpose must be 'training' or 'inference'")
