import argparse
from pathlib import Path

import yaml

from vq_sce.inference import Inference, MultiScaleInference, SingleScaleInference
from vq_sce.networks.model import Task


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", "-d", help="Data path", type=str)
    parser.add_argument("--original", "-o", help="Original data path", type=str)
    parser.add_argument("--stage", "-s", help="Joint stage", type=str)
    parser.add_argument("--minibatch", "-m", help="Minibatch size", type=int, default=4)
    parser.add_argument("--option", "-op", help="`save`, `display`", type=str)
    parser.add_argument("--dev", "-dv", help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    # Parse config json
    with open(expt_path / "config.yml") as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = Path(arguments.path)
    config["data"]["data_path"] = Path(arguments.data) / "test"
    config["expt"]["mb_size"] = arguments.minibatch

    # Optional path to original NRRD data (for saving metadata with predictions)
    if arguments.original is None:
        config["paths"]["original_path"] = None
    else:
        config["paths"]["original_path"] = Path(arguments.original)

    # Development mode if necessary
    if arguments.dev:
        dims = config["data"]["source_dims"]
        config["data"]["source_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        dims = config["data"]["target_dims"]
        config["data"]["target_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        config["data"]["down_sample"] = 4
    else:
        config["data"]["down_sample"] = 1

    # Set data type
    if config["expt"]["expt_type"] == Task.JOINT:
        config["data"]["type"] = arguments.stage

    inference: Inference

    if len(config["hyperparameters"]["scales"]) == 1:
        inference = SingleScaleInference(config, arguments.stage)
    else:
        inference = MultiScaleInference(config, arguments.stage)

    if config["expt"]["expt_type"] == Task.JOINT:
        config["data"]["type"] = arguments.stage
    else:
        assert arguments.stage == config["data"]["type"]

    inference.run(arguments.option)


if __name__ == "__main__":
    main()
