import argparse
import datetime
import os
from pathlib import Path
import tensorflow as tf
import yaml

from vq_sce import RANDOM_SEED
from vq_sce.networks.build_model import build_model
from vq_sce.trainers.build_trainer import build_training_loop


#-------------------------------------------------------------------------

def train(config: dict, dev: bool):
    tf.random.set_seed(RANDOM_SEED)
    tf.get_logger().setLevel("ERROR")

    # Development mode if necessary
    if dev:
        dims = config["data"]["source_dims"]
        config["data"]["source_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        dims = config["data"]["target_dims"]
        config["data"]["target_dims"] = [dims[0], dims[1] // 4, dims[2] // 4]
        config["data"]["down_sample"] = 4

    # Get model
    model = build_model(config)

    if config["expt"]["verbose"]:
        model.summary()

    # Write graph for visualising in Tensorboard
    if config["expt"]["graph"]:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = str(Path(config['paths']['expt_path']) / "logs" / curr_time)
        writer = tf.summary.create_file_writer(log_dir)

        @tf.function
        def trace(x):
            model(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + config["data"]["source_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    training_loop = build_training_loop(config, model, dev)

    # Run training loop
    training_loop.train()


#-------------------------------------------------------------------------

def main():

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    parser.add_argument("--dev", "-d", help="Development mode", action="store_true")
    arguments = parser.parse_args()

    expt_path = Path(arguments.path)

    if not os.path.exists(expt_path / "images"):
        os.makedirs(expt_path / "images")

    if not os.path.exists(expt_path / "logs"):
        os.makedirs(expt_path / "logs")

    if not os.path.exists(expt_path / "models"):
        os.makedirs(expt_path / "models")

    # Parse config json
    with open(expt_path / "config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)
    
    config["paths"]["expt_path"] = arguments.path

    # Set GPU
    if arguments.gpu is not None:
        gpu_number = arguments.gpu
        os.environ["LD_LIBRARY_PATH"] = config["paths"]["cuda_path"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)
    
    train(config, arguments.dev)


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
