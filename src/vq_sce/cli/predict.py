import argparse
from pathlib import Path
import yaml

from vq_sce.inference import Inference, SingleScaleInference, MultiScaleInference


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--minibatch", '-m', help="Minibatch size", type=int, default=128)
    parser.add_argument("--save", '-s', help="Save images", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    # Parse config json
    with open(expt_path / "config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    config["paths"]["expt_path"] = Path(arguments.path)
    config["data"]["data_path"] = Path(arguments.data)
    config["expt"]["mb_size"] = arguments.minibatch

    inference: Inference

    if len(config["hyperparameters"]["scales"]) > 1:
        inference = SingleScaleInference(config)
    else:
        inference = MultiScaleInference(config)

    inference.run(arguments.save)


if __name__ == "__main__":
    main()
