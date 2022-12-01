import argparse
import datetime
import os
from pathlib import Path
import tensorflow as tf
import yaml

from vq_sce import RANDOM_SEED
from vq_sce.networks.build_model import build_model
from vq_sce.trainingloops.training_loop import TrainingLoop
from vq_sce.utils.build_dataloader import get_train_dataloader


#-------------------------------------------------------------------------

def train(config: dict):
    tf.random.set_seed(RANDOM_SEED)
    tf.get_logger().setLevel("ERROR")

    # Get datasets and data generator
    train_ds, val_ds, train_gen, val_gen = get_train_dataloader(config)

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
            if config["data"]["times"] is not None:
                return model.UNet(x, 0.0)
            else:
                return model.UNet(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + config["hyperparameters"]["img_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    training_loop = TrainingLoop(Model=model,
                                 dataset=(train_ds, val_ds),
                                 train_generator=train_gen,
                                 val_generator=val_gen,
                                 config=config)

    # Run training loop
    training_loop.train()


#-------------------------------------------------------------------------

def main():

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
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
    
    train(config)


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
