import argparse
import datetime
import os
from pathlib import Path
import sys
import tensorflow as tf
import yaml

from vec_quant_sCE.networks.model import Model
from vec_quant_sCE.trainingloops.training_loop import TrainingLoop
from vec_quant_sCE.utils.build_dataloader import get_train_dataloader


#-------------------------------------------------------------------------

def train(CONFIG):
    tf.random.set_seed(5)
    tf.get_logger().setLevel("ERROR")

    # Get datasets and data generator
    train_ds, val_ds, train_gen, val_gen = get_train_dataloader(CONFIG)

    # Compile model
    #model = Model(CONFIG)
    source = tf.keras.Input(shape=[64, 64, 64, 1])
    up = tf.keras.layers.UpSampling3D(size=(2, 2, 1))
    pred = up(source)
    model = tf.keras.Model(inputs=source, outputs=[pred, source])
    optimiser = tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["opt"], name="opt")
    model.compile(optimiser)

    if CONFIG["expt"]["verbose"]:
        model.summary()

    # Write graph for visualising in Tensorboard
    if CONFIG["expt"]["graph"]:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = str(Path(CONFIG['paths']['expt_path']) / "logs" / curr_time)
        writer = tf.summary.create_file_writer(log_dir)

        @tf.function
        def trace(x):
            if CONFIG["data"]["times"] is not None:
                return model.UNet(x, 0.0)
            else:
                return model.UNet(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + CONFIG["hyperparameters"]["img_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    training_loop = TrainingLoop(Model=model,
                                 dataset=(train_ds, val_ds),
                                 train_generator=train_gen,
                                 val_generator=val_gen,
                                 config=CONFIG)

    # Run training loop
    training_loop.train()


#-------------------------------------------------------------------------

def main(args=None):

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    if not os.path.exists(Path(EXPT_PATH) / "images"):
        os.makedirs(Path(EXPT_PATH) / "images")

    if not os.path.exists(Path(EXPT_PATH) / "logs"):
        os.makedirs(Path(EXPT_PATH) / "logs")

    if not os.path.exists(Path(EXPT_PATH) / "models"):
        os.makedirs(Path(EXPT_PATH) / "models")

    # Parse config json
    with open(Path(EXPT_PATH) / "config.yml", 'r') as infile:
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


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv[1:])
