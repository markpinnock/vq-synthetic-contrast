import glob
from pathlib import Path
import tensorflow as tf

from vq_sce.utils.dataloaders.contrast_dataloader import ContrastDataloader
from vq_sce.utils.dataloaders.super_res_dataloader import SuperResDataloader

DATALOADER_DICT = {
    "contrast": ContrastDataloader,
    "super_res": SuperResDataloader
}

INFERENCE_MB_SIZE = 1


def get_train_dataloader(config: dict):

    # Specify output types
    output_types = ["source", "target"]

    if config["data"]["segs"] is not None:
        output_types += ["seg"]
    else:
        assert config["hyperparameters"]["mu"] == 0.0
    
    if config["data"]["times"] is not None:
        output_types += ["times"]

    try:
        config["data"]["scales"] = config["hyperparameters"]["scales"]
    except KeyError:
        config["data"]["scales"] = [8]

    # Initialise datasets and set normalisation parameters
    Dataloader = DATALOADER_DICT[config["data"]["type"]]
    TrainGenerator = Dataloader(config=config["data"], dataset_type="training")
    ValGenerator = Dataloader(config=config["data"], dataset_type="validation")

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
    config: dict,
    by_subject: bool = False,
    stride_length: int = None
):

    # Inference-specific config settings
    config["data"]["cv_folds"] = 1
    config["data"]["fold"] = 0
    config["data"]["segs"] = None
    config["data"]["xy_patch"] = True
    config["data"]["stride_length"] = stride_length

    # Initialise datasets and set normalisation parameters
    Dataloader = DATALOADER_DICT[config["data"]["type"]]
    TestGenerator = Dataloader(config=config["data"], dataset_type="validation")

    # Create dataloader
    if by_subject:
        # Specify output types
        output_types = {
            "source": "float32",
            "subject_ID": tf.string,
            "coords": "int32"
        }

        data_path = config["data"]["data_path"]
        source_list = glob.glob(str(Path(data_path) / "Images" / "*HQ*"))
        source_list = [f[-15:] for f in source_list]

        subject_datasets = {}

        for source in source_list:
            test_ds = tf.data.Dataset.from_generator(
                generator=TestGenerator.subject_generator,
                args=[source],
                output_types=output_types).batch(config["expt"]["mb_size"])
            subject_datasets[source[:-4]] = test_ds

        return subject_datasets, TestGenerator

    else:
        output_types = {"source": "float32", "subject_id": tf.string}

        test_ds = tf.data.Dataset.from_generator(
            generator=TestGenerator.inference_generator,
            output_types=output_types).batch(INFERENCE_MB_SIZE)

        return test_ds, TestGenerator
