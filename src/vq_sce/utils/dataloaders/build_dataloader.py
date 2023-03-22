import glob
from pathlib import Path
import tensorflow as tf
from typing import Any

from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader
from vq_sce.utils.dataloaders.contrast_dataloader import ContrastDataloader
from vq_sce.utils.dataloaders.super_res_dataloader import SuperResDataloader

DATALOADER_DICT = {
    "contrast": ContrastDataloader,
    "super_res": SuperResDataloader
}
DataloaderType = tuple[
    tf.data.Dataset,
    BaseDataloader,
    tf.data.Dataset,
    BaseDataloader,
]
INFERENCE_MB_SIZE = 1


def get_train_dataloader(config: dict[str, Any], dev: bool) -> DataloaderType:

    # Specify output types and scale
    output_types = ["source", "target"]

    # Initialise datasets and set normalisation parameters
    Dataloader = DATALOADER_DICT[config["data"]["type"]]
    TrainGenerator = Dataloader(config=config["data"], dataset_type="training", dev=dev)
    ValGenerator = Dataloader(config=config["data"], dataset_type="validation", dev=dev)

    # Create dataloader
    train_ds = tf.data.Dataset.from_generator(
        generator=TrainGenerator.data_generator,
        output_types={k: "float32" for k in output_types}
        ).batch(config["expt"]["mb_size"])

    val_ds = tf.data.Dataset.from_generator(
        generator=ValGenerator.data_generator,
        output_types={k: "float32" for k in output_types}
        ).batch(config["expt"]["mb_size"])

    return train_ds, val_ds, TrainGenerator, ValGenerator


#-------------------------------------------------------------------------


def get_test_dataloader(
    config: dict[str, Any],
    dev: bool = False,
) -> tuple[tf.data.Dataset, BaseDataloader]:

    # Inference-specific config settings
    config["data"]["cv_folds"] = 1
    config["data"]["fold"] = 0

    # Initialise datasets and set normalisation parameters
    Dataloader = DATALOADER_DICT[config["data"]["type"]]
    TestGenerator = Dataloader(config=config["data"], dataset_type="validation", dev=dev)

    # Create dataloader
    output_types = {"source": "float32", "subject_id": tf.string}

    test_ds = tf.data.Dataset.from_generator(
        generator=TestGenerator.inference_generator,
        output_types=output_types).batch(INFERENCE_MB_SIZE)

    return test_ds, TestGenerator
