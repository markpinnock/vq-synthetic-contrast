import argparse
import os
from pathlib import Path
from typing import Any

import absl.logging
import tensorflow as tf
import yaml

from vq_sce import RANDOM_SEED
from vq_sce.callbacks.build_callbacks import build_callbacks_and_datasets
from vq_sce.networks.build_model import build_model_train

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

    # Set distributed training strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"\nUsing {strategy.num_replicas_in_sync} devices\n")  # noqa: T201
    config["expt"]["local_mb_size"] = config["expt"]["mb_size"]
    config["expt"]["mb_size"] *= strategy.num_replicas_in_sync
    config["hyperparameters"]["opt"]["learning_rate"] *= strategy.num_replicas_in_sync
    if "alpha_opt" in config["hyperparameters"].keys():
        config["hyperparameters"]["alpha_opt"][
            "learning_rate"
        ] *= strategy.num_replicas_in_sync

    # Get model
    model = build_model_train(config, strategy=strategy, dev=dev)

    if config["expt"]["verbose"]:
        model.summary()

    callbacks_and_datasets = build_callbacks_and_datasets(config, dev=dev)

    # If DARTS model, need to supply both train and validation data in training
    if config["expt"]["optimisation_type"] == "DARTS":
        train_ds = callbacks_and_datasets["train_ds"]
        valid_ds = callbacks_and_datasets["valid_ds"]
        callbacks_and_datasets["train_ds"] = tf.data.Dataset.zip((train_ds, valid_ds))

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
    parser.add_argument("--gpu", "-g", help="GPU number", type=str)
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
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if arguments.gpu is not None:
        gpu_numbers = [int(gpu) for gpu in arguments.gpu.split(",")]
        os.environ["LD_LIBRARY_PATH"] = config["paths"]["cuda_path"]
        gpus = tf.config.list_physical_devices("GPU")

        if arguments.dev:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=15360 // len(gpu_numbers),
                    )
                    for _ in gpu_numbers
                ],
            )

        else:
            visible_gpus = []
            for gpu_number in gpu_numbers:
                visible_gpus.append(gpus[gpu_number])
            tf.config.set_visible_devices(visible_gpus, "GPU")
            for gpu in visible_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    train(config, arguments.dev)


# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
