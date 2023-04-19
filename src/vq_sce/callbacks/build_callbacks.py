from pathlib import Path
from typing import Any

import tensorflow as tf

from vq_sce.callbacks.callbacks import SaveExamples, SaveResults
from vq_sce.utils.dataloaders.build_dataloader import get_train_dataloader
from vq_sce.utils.dataloaders.joint_dataset import JointDataset


def build_callbacks_and_datasets(
    config: dict[str, Any],
    model: tf.keras.Model,
    dev: bool,
) -> dict[str, Any]:
    expt_type = config["expt"]["expt_type"]

    # Get datasets and data generators
    if expt_type == "single":
        train_ds, valid_ds, train_gen, valid_gen = get_train_dataloader(config, dev)

    elif expt_type == "joint":
        config["data"]["type"] = "contrast"
        ce_train_ds, ce_valid_ds, _, _ = get_train_dataloader(config, dev)
        config["data"]["type"] = "super_res"
        sr_train_ds, sr_valid_ds, _, _ = get_train_dataloader(config, dev)

        train_ds = tf.data.Dataset.zip({"ce": ce_train_ds, "sr": sr_train_ds})
        valid_ds = tf.data.Dataset.zip({"ce": ce_valid_ds, "sr": sr_valid_ds})

        train_gen = JointDataset(config=config["data"], dataset_type="training")
        valid_gen = JointDataset(config=config["data"], dataset_type="validation")

    else:
        raise ValueError("Must be `single` or `joint`")

    # Get callbacks
    expt_path = Path(config["paths"]["expt_path"])
    save_freq = config["expt"]["save_every"]

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=expt_path / "models",
        save_weights_only=False,
        save_freq=save_freq,
    )

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
        data_type=config["data"]["data_type"],
    )

    save_examples = SaveExamples(
        filepath=expt_path / "images",
        save_freq=save_freq,
        train_generator=train_gen,
        valid_generator=valid_gen,
    )

    callbacks = [model_checkpoint, tensorboard, save_results, save_examples]

    return {"train_ds": train_ds, "valid_ds": valid_ds, "callbacks": callbacks}
