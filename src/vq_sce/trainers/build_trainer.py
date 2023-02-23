import tensorflow as tf

from .trainer import TrainingLoop
from .joint_trainer import JointTrainingLoop
from vq_sce.utils.dataloaders.build_dataloader import get_train_dataloader
from vq_sce.utils.dataloaders.joint_dataset import JointDataset

TRAINER_DICT = {
    "single": TrainingLoop,
    "joint": JointTrainingLoop
}


def build_training_loop(config: dict, model: tf.keras.Model, dev: bool):
    expt_type = config["expt"]["expt_type"]

    # Get datasets and data generator
    if expt_type == "single":
        train_ds, val_ds, train_gen, val_gen = get_train_dataloader(config, dev)
        datasets = {
            "train_dataset": train_ds,
            "val_dataset": val_ds,
            "train_generator": train_gen,
            "val_generator": val_gen
        }

    elif expt_type == "joint":
        config["data"]["type"] = "contrast"
        ce_train_ds, ce_val_ds, _, _ = get_train_dataloader(config, dev)
        config["data"]["type"] = "super_res"
        sr_train_ds, sr_val_ds, _, _ = get_train_dataloader(config, dev)

        train_gen = JointDataset(config=config["data"], dataset_type="training")
        val_gen = JointDataset(config=config["data"], dataset_type="validation")
        datasets = {
            "sr_train_dataset": sr_train_ds,
            "sr_val_dataset": sr_val_ds,
            "ce_train_dataset": ce_train_ds,
            "ce_val_dataset": ce_val_ds,
            "train_generator": train_gen,
            "val_generator": val_gen
        }

    else:
        raise ValueError("Must be `single` or `joint`")

    training_loop = TRAINER_DICT[expt_type](Model=model,
                                            config=config,
                                            **datasets)

    return training_loop
