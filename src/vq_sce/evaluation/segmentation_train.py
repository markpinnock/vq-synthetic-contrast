import argparse
import os
import tensorflow as tf
import yaml

from .unet import UNet
from .util import load_data, Augmentation, dice_loss, ImgSaveCallback

#-------------------------------------------------------------------------

def train(expt_path, save_images, config):
    tf.random.set_seed(5)

    time_str = str(config["data_config"]["img_type"]).replace('.', '_')

    train_ds = tf.data.Dataset.from_tensor_slices(
        load_data("train", config["model_config"]["img_dims"][0] + 12, **config["data_config"])).shuffle(256)

    train_examples = next(iter(train_ds.batch(4).map(Augmentation(config["model_config"]["img_dims"]))))
    train_ds = train_ds.batch(config["model_config"]["mb_size"]).map(Augmentation(config["model_config"]["img_dims"]))

    test_ds = tf.data.Dataset.from_tensor_slices(
        load_data("test", config["model_config"]["img_dims"][0], **config["data_config"]))

    test_examples = next(iter(test_ds.shuffle(256).batch(4)))
    test_ds = test_ds.batch(config["model_config"]["mb_size"])

    Model = UNet(config["model_config"])
    if not os.path.exists(f"{expt_path}/{time_str}/logs"): os.makedirs(f"{expt_path}/{time_str}/logs")
    if not os.path.exists(f"{expt_path}/{time_str}/models"): os.makedirs(f"{expt_path}/{time_str}/models")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{expt_path}/{time_str}/logs", write_graph=False)
    img_save_callback = ImgSaveCallback(train_examples, test_examples, f"{expt_path}/images")

    inputs = tf.keras.Input(shape=config["model_config"]["img_dims"] + [1])
    outputs = Model.call(inputs, training=True)
    tf.keras.Model(inputs=inputs, outputs=outputs).summary()

    callbacks = [tensorboard_callback]
    if save_images: callbacks += [img_save_callback]

    Model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=dice_loss)
    Model.fit(
        train_ds,
        epochs=config["model_config"]["epochs"],
        validation_data=test_ds,
        callbacks=callbacks
        )
    Model.save_weights(f"{expt_path}/{time_str}/models/model.ckpt")


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Segmentation-based evaluation training routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--time", '-t', type=float)
    parser.add_argument("--save_images", '-s', action="store_true")
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)

    if arguments.time is not None:
        CONFIG["data_config"]["img_type"] = arguments.time

    train(EXPT_PATH, arguments.save_images, CONFIG)
