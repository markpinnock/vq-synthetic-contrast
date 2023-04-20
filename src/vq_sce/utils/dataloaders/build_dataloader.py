import enum
from typing import Any

import tensorflow as tf

from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader
from vq_sce.utils.dataloaders.contrast_dataloader import ContrastDataloader
from vq_sce.utils.dataloaders.super_res_dataloader import SuperResDataloader

DATALOADER_DICT = {"contrast": ContrastDataloader, "super_res": SuperResDataloader}
DataloaderType = tuple[
    tf.data.Dataset,
    BaseDataloader,
    tf.data.Dataset,
    BaseDataloader,
]
INFERENCE_MB_SIZE = 1


# -------------------------------------------------------------------------


@enum.unique
class Subsets(str, enum.Enum):
    TRAIN = "training"
    VALID = "validation"
    TEST = "test"


# -------------------------------------------------------------------------


def get_train_dataloader(config: dict[str, Any], dev: bool) -> DataloaderType:
    # Specify output types and scale
    output_types = ["source", "target"]

    # Initialise datasets and set normalisation parameters
    dataloader = DATALOADER_DICT[config["data"]["type"]]
    train_generator = dataloader(
        config=config["data"],
        dataset_type=Subsets.TRAIN,
        dev=dev,
    )
    val_generator = dataloader(
        config=config["data"],
        dataset_type=Subsets.VALID,
        dev=dev,
    )

    # Create dataloader
    train_ds = (
        tf.data.Dataset.from_generator(
            generator=train_generator.data_generator,
            output_types={k: "float32" for k in output_types},
        )
        .repeat()
        .batch(config["expt"]["mb_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_generator(
            generator=val_generator.data_generator,
            output_types={k: "float32" for k in output_types},
        )
        .repeat()
        .batch(config["expt"]["mb_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, train_generator, val_generator


# -------------------------------------------------------------------------


def get_test_dataloader(
    config: dict[str, Any],
    subset: str = Subsets.TEST,
    dev: bool = False,
) -> tuple[tf.data.Dataset, BaseDataloader]:
    dataloader = DATALOADER_DICT[config["data"]["type"]]

    if subset == Subsets.TEST:
        # Test-specific config settings
        config["data"]["cv_folds"] = 1
        config["data"]["fold"] = 0

    if subset == Subsets.TRAIN:
        test_generator = dataloader(
            config=config["data"],
            dataset_type=Subsets.TRAIN,
            dev=dev,
        )

    else:
        test_generator = dataloader(
            config=config["data"],
            dataset_type=Subsets.VALID,
            dev=dev,
        )

    # Create dataloader
    output_types = {
        "source": "float32",
        "target": "float32",
        "source_id": tf.string,
        "target_id": tf.string,
    }

    test_ds = tf.data.Dataset.from_generator(
        generator=test_generator.inference_generator,
        output_types=output_types,
    ).batch(INFERENCE_MB_SIZE)

    return test_ds, test_generator
