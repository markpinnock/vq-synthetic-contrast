import argparse
import datetime
import os
import tensorflow as tf
import yaml

from vec_quant_sCE.networks.unet import UNet
from vec_quant_sCE.trainingloops.training_loop import TrainingLoop
from vec_quant_sCE.utils.build_dataloader import get_train_dataloader


#-------------------------------------------------------------------------

def train(CONFIG):

    # Get datasets and data generator
    train_ds, val_ds, train_gen, val_gen = get_train_dataloader(CONFIG)

    # Compile model
    Model = UNet(CONFIG)

    if CONFIG["expt"]["verbose"]:
        Model.summary()

    # Write graph for visualising in Tensorboard
    if CONFIG["expt"]["graph"]:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{CONFIG['paths']['expt_path']}/logs/{curr_time}"
        writer = tf.summary.create_file_writer(log_dir)

        @tf.function
        def trace(x):
            return Model.Generator(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + CONFIG["hyperparameters"]["img_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    training_loop = TrainingLoop(Model=Model,
                                 dataset=(train_ds, val_ds),
                                 train_generator=train_gen,
                                 val_generator=val_gen,
                                 config=CONFIG)

    # Run training loop
    training_loop.train()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Training routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    if not os.path.exists(f"{EXPT_PATH}/images"):
        os.makedirs(f"{EXPT_PATH}/images")

    if not os.path.exists(f"{EXPT_PATH}/logs"):
        os.makedirs(f"{EXPT_PATH}/logs")

    if not os.path.exists(f"{EXPT_PATH}/models"):
        os.makedirs(f"{EXPT_PATH}/models")

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path

    # Set GPU
    if arguments.gpu is not None:
        gpu_number = arguments.gpu
        os.environ["LD_LIBRARY_PATH"] = CONFIG["paths"]["cuda_path"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)
    
    train(CONFIG)
