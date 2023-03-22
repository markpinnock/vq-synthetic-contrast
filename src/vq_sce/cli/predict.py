import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import SimpleITK as itk
import tensorflow as tf
from typing import Any
import yaml



#-------------------------------------------------------------------------



#-------------------------------------------------------------------------

def main():
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

    if config["hyperparameters"]["scales"] == [1]:
        full_size_inference(config, arguments.save)
    elif config["hyperparameters"]["scales"] == [4]:
        patch_inference(config, arguments.save)
    elif len(config["hyperparameters"]["scales"]) > 1:
        multiscale_inference(config, arguments.save)
    else:
        raise ValueError(f"Scales not recognised: {config['hyperparameters']['scales']}")


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
