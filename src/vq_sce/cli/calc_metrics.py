import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from vq_sce.inference import Inference, MultiScaleInference, SingleScaleInference
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.build_dataloader import Subsets, get_test_dataloader

METRICS = ["L1", "MSE", "pSNR", "SSIM"]


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", "-d", help="Data path", type=str)
    parser.add_argument("--stage", "-st", help="Joint stage", type=str)
    parser.add_argument("--subset", "-su", help="Data subset", type=str)
    parser.add_argument("--epoch", "-ep", help="Model save epoch", type=str)
    parser.add_argument("--minibatch", "-m", help="Minibatch size", type=int, default=8)
    parser.add_argument("--dev", "-dv", help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    # Parse config json
    with open(expt_path / "config.yml") as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = Path(arguments.path)
    config["paths"]["original_path"] = None
    config["data"]["data_path"] = Path(arguments.data) / "test"
    config["expt"]["mb_size"] = arguments.minibatch

    # Development mode if necessary
    if arguments.dev:
        dims = config["data"]["source_dims"]
        config["data"]["source_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        dims = config["data"]["target_dims"]
        config["data"]["target_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        config["data"]["down_sample"] = 4
    else:
        config["data"]["down_sample"] = 1

    config_copy = copy.deepcopy(
        config,
    )  # Avoid subset-specific params being overwritten
    inference: Inference

    if len(config["hyperparameters"]["scales"]) == 1:
        inference = SingleScaleInference(
            config_copy,
            stage=arguments.stage,
            epoch=arguments.epoch,
        )
    else:
        inference = MultiScaleInference(
            config_copy,
            stage=arguments.stage,
            epoch=arguments.epoch,
        )

    if arguments.subset is None:
        subsets = Subsets
    else:
        assert arguments.subset in list(Subsets), (arguments.subset, list(Subsets))
        subsets = [arguments.subset]  # type: ignore

    for subset in subsets:
        config_copy = copy.deepcopy(
            config,
        )  # Avoid subset-specific params being overwritten

        if subset == Subsets.TEST:
            config_copy["data"]["data_path"] = Path(arguments.data) / "test"
        else:
            config_copy["data"]["data_path"] = Path(arguments.data) / "train"

        if config_copy["expt"]["expt_type"] == Task.JOINT:
            config_copy["data"]["type"] = arguments.stage
        else:
            assert arguments.stage == config_copy["data"]["type"]

        # Set up dataset and override Inference class's default one
        test_ds, test_generator = get_test_dataloader(config_copy, subset=subset)
        inference.test_ds = test_ds
        inference.TestGenerator = test_generator
        inference.data_path = config_copy["data"]["data_path"]

        metric_dict = inference.run(option="metrics")

        for metric in METRICS:
            csv_name = f"{arguments.stage}_{subset}_{metric}"

            # Create dataframe if not present
            df_path = config_copy["paths"]["expt_path"].parent / f"{csv_name}.csv"

            try:
                df = pd.read_csv(df_path, index_col=0)

            except FileNotFoundError:
                df = pd.DataFrame(index=metric_dict["id"])
                df[
                    f"{config_copy['paths']['expt_path'].stem}_{arguments.epoch}"
                ] = metric_dict[metric]
            else:
                new_df = pd.DataFrame(
                    metric_dict[metric],
                    index=metric_dict["id"],
                    columns=[
                        f"{config_copy['paths']['expt_path'].stem}_{arguments.epoch}",
                    ],
                )
                df = df.join(new_df, how="outer")

            df.to_csv(df_path, index=True)


if __name__ == "__main__":
    main()
