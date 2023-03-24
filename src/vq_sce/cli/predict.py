import argparse
from pathlib import Path
import yaml

from vq_sce.inference import Inference, SingleScaleInference, MultiScaleInference


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--original", '-o', help="Original data path", type=str)
    parser.add_argument("--minibatch", '-m', help="Minibatch size", type=int, default=4)
    parser.add_argument("--option", '-op', help="`save`, `display` or `metrics`", type=str)
    parser.add_argument("--dev", '-dv', help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    # Parse config json
    with open(expt_path / "config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = Path(arguments.path)
    config["data"]["data_path"] = Path(arguments.data)
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

    inference: Inference

    if len(config["hyperparameters"]["scales"]) == 1:
        inference = SingleScaleInference(config)
    else:
        inference = MultiScaleInference(config)

    inference.run(arguments.option)


if __name__ == "__main__":
    main()
