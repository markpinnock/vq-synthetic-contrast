from pathlib import Path
from typing import Any

import tensorflow as tf

from vq_sce.callbacks.callbacks import SaveExamples, SaveModel, SaveResults
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.build_dataloader import get_train_dataloader
from vq_sce.utils.dataloaders.joint_dataset import JointDataset


def build_callbacks_and_datasets(config: dict[str, Any], dev: bool) -> dict[str, Any]:
    expt_type = config["expt"]["expt_type"]

    # Get datasets and data generators
    if expt_type == "single":
        train_ds, valid_ds, train_gen, valid_gen = get_train_dataloader(config, dev)
        train_examples = len(train_gen.data["source"])
        valid_examples = len(valid_gen.data["source"])
        train_steps = train_examples // config["expt"]["mb_size"]
        valid_steps = valid_examples // config["expt"]["mb_size"]

    elif expt_type == "joint":
        config["data"]["type"] = Task.CONTRAST
        ce_train_ds, ce_valid_ds, ce_train_gen, ce_valid_gen = get_train_dataloader(
            config,
            dev,
        )
        config["data"]["type"] = Task.SUPER_RES
        sr_train_ds, sr_valid_ds, sr_train_gen, sr_valid_gen = get_train_dataloader(
            config,
            dev,
        )

        if len(ce_train_gen.data["source"]) < len(sr_train_gen.data["source"]):
            train_examples = len(ce_train_gen.data["source"])
        else:
            train_examples = len(sr_train_gen.data["source"])
        train_steps = train_examples // config["expt"]["mb_size"]

        if len(ce_valid_gen.data["source"]) < len(sr_valid_gen.data["source"]):
            valid_examples = len(ce_valid_gen.data["source"])
        else:
            valid_examples = len(sr_valid_gen.data["source"])
        valid_steps = valid_examples // config["expt"]["mb_size"]

        train_ds = tf.data.Dataset.zip({"ce": ce_train_ds, "sr": sr_train_ds})
        valid_ds = tf.data.Dataset.zip({"ce": ce_valid_ds, "sr": sr_valid_ds})

        train_gen = JointDataset(config=config["data"], dataset_type="training")
        valid_gen = JointDataset(config=config["data"], dataset_type="validation")

    else:
        raise ValueError("Must be `single` or `joint`")

    # Get callbacks
    expt_path = Path(config["paths"]["expt_path"])
    save_freq = config["expt"]["save_every"]

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=expt_path / "logs",
        histogram_freq=int(config["expt"]["log_histograms"]),
        write_graph=config["expt"]["verbose"],
        write_images=config["expt"]["verbose"],
        update_freq="epoch",
    )

    save_results = SaveResults(
        filepath=expt_path / "logs",
        save_freq=save_freq,
        data_type=config["data"]["type"],
    )

    model_checkpoint = SaveModel(
        filepath=expt_path / "models",
        save_weights_only=False,
        save_freq=save_freq,
    )

    save_examples = SaveExamples(
        filepath=expt_path / "images",
        save_freq=save_freq,
        train_generator=train_gen,
        valid_generator=valid_gen,
    )

    callbacks = [model_checkpoint, tensorboard, save_results, save_examples]

    return {
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "callbacks": callbacks,
        "train_steps": train_steps,
        "valid_steps": valid_steps,
    }
