import tensorflow as tf

from .model import Model
from .multiscale_model import MultiscaleModel

MODEL_DICT = {
    "single_scale": Model,
    "multi_scale": MultiscaleModel
}


def build_model(config: dict, purpose: str = "training"):
    if "scales" not in config["hyperparameters"].keys():
        model_type = "single_scale"
    elif len(config["hyperparameters"]["scales"]) == 1:
        model_type = "single_scale"
    else:
        model_type = "multi_scale"

    model = MODEL_DICT[model_type](config)

    if purpose == "training":
        optimiser = tf.keras.optimizers.Adam(*config["hyperparameters"]["opt"], name="opt")
        model.compile(optimiser)
        return model
    elif purpose == "inference":
        _ = model(tf.zeros([1] + config["data"]["patch_size"] + [1]))
        model.UNet.load_weights(config["paths"]["expt_path"] / "models" / "model.ckpt")
        return model
    else:
        raise ValueError("Purpose must be 'training' or 'inference'")
