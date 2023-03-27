import argparse
import copy
import pandas as pd
from pathlib import Path
import yaml

from vq_sce.inference import Inference, SingleScaleInference, MultiScaleInference
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.build_dataloader import Subsets, get_test_dataloader

METRICS = ["L1"]


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--joint_stage", '-j', help="Joint stage", type=str)
    parser.add_argument("--minibatch", '-m', help="Minibatch size", type=int, default=1)
    parser.add_argument("--dev", '-dv', help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    # Parse config json
    with open(expt_path / "config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = Path(arguments.path)
    config["paths"]["original_path"] = None
    config["data"]["data_path"] = Path(arguments.data) / "test"
    config["expt"]["mb_size"] = arguments.minibatch

    if config["expt"]["expt_type"] == "single":
        config["expt"]["expt_type"] = config["data"]["type"]

    # Development mode if necessary
    if arguments.dev:
        dims = config["data"]["source_dims"]
        config["data"]["source_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        dims = config["data"]["target_dims"]
        config["data"]["target_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        config["data"]["down_sample"] = 4
    else:
        config["data"]["down_sample"] = 1

    config_copy = copy.deepcopy(config)  # Avoid subset-specific params being overwritten
    inference: Inference

    if len(config["hyperparameters"]["scales"]) == 1:
        inference = SingleScaleInference(config_copy, joint_stage=arguments.joint_stage)
    else:
        inference = MultiScaleInference(config_copy, joint_stage=arguments.joint_stage)

    for subset in Subsets:
        config_copy = copy.deepcopy(config)  # Avoid subset-specific params being overwritten

        if subset == Subsets.TEST:
            config_copy["data"]["data_path"] = Path(arguments.data) / "test"
        else:
            config_copy["data"]["data_path"] = Path(arguments.data) / "train"

        if config_copy["expt"]["expt_type"] == Task.JOINT:
            assert arguments.joint_stage in [Task.CONTRAST, Task.SUPER_RES]
            config_copy["data"]["type"] = arguments.joint_stage

        # Set up dataset and override Inference class's default one
        test_ds, TestGenerator = get_test_dataloader(config_copy, subset=subset)
        inference.test_ds = test_ds
        inference.TestGenerator = TestGenerator
        inference.data_path = config_copy["data"]["data_path"]

        metric_dict = inference.run(option="metrics")

        for metric in METRICS:
            if config_copy["expt"]["expt_type"] == Task.JOINT:
                csv_name = f"{arguments.joint_stage}_{subset}_{metric}"
            else:
                csv_name = f"{config_copy['data']['type']}_{subset}_{metric}"

            # Create dataframe if not present
            df_path = config_copy["paths"]["expt_path"].parent / f"{csv_name}.csv"

            try:
                df = pd.read_csv(df_path, index_col=0)

            except FileNotFoundError:
                df = pd.DataFrame(index=metric_dict["id"])
                df[f"{config_copy['paths']['expt_path'].stem}"] = metric_dict[metric]

            else:
                new_df = pd.DataFrame(metric_dict["L1"], index=metric_dict["id"], columns=[f"{config_copy['paths']['expt_path'].stem}"])
                df = df.join(new_df, how="outer")

            df.to_csv(df_path, index=True)


if __name__ == "__main__":
    main()
