import argparse
import os
from pathlib import Path
from typing import Any

import absl.logging
import tensorflow as tf
import yaml

from vq_sce import RANDOM_SEED
from vq_sce.callbacks.build_callbacks import build_callbacks_and_datasets
from vq_sce.networks.build_model import build_model

# -------------------------------------------------------------------------


def train(config: dict[str, Any], dev: bool) -> None:
    tf.random.set_seed(RANDOM_SEED)
    absl.logging.set_verbosity(absl.logging.ERROR)

    # Development mode if necessary
    if dev:
        dims = config["data"]["source_dims"]
        config["data"]["source_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        dims = config["data"]["target_dims"]
        config["data"]["target_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        config["data"]["down_sample"] = 4

    else:
        config["data"]["down_sample"] = 1

    # Get model
    model = build_model(config)

    if config["expt"]["verbose"]:
        model.summary()

    callbacks_and_datasets = build_callbacks_and_datasets(config, dev)

    # Run training
    model.fit(
        x=callbacks_and_datasets["train_ds"],
        epochs=config["expt"]["epochs"],
        callbacks=callbacks_and_datasets["callbacks"],
        validation_data=callbacks_and_datasets["valid_ds"],
        initial_epoch=0,
        steps_per_epoch=callbacks_and_datasets["train_steps"],
        validation_steps=callbacks_and_datasets["valid_steps"],
    )


# -------------------------------------------------------------------------


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    parser.add_argument("--dev", "-d", help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    if not os.path.exists(expt_path / "images"):
        os.makedirs(expt_path / "images")

    if not os.path.exists(expt_path / "logs"):
        os.makedirs(expt_path / "logs")

    if not os.path.exists(expt_path / "models"):
        os.makedirs(expt_path / "models")

    # Parse config json
    with open(expt_path / "config.yml") as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = arguments.path

    # Set GPU
    if arguments.gpu is not None:
        gpu_number = arguments.gpu
        os.environ["LD_LIBRARY_PATH"] = config["paths"]["cuda_path"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)

    train(config, arguments.dev)


# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
