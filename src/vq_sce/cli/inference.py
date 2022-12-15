import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import SimpleITK as itk
import tensorflow as tf
import yaml

from vq_sce.networks.build_model import build_model
from vq_sce.utils.dataloaders.build_dataloader import get_test_dataloader
from vq_sce.utils.combine_patches import CombinePatches
from vq_sce.utils.patch_utils import generate_indices, extract_patches

STRIDE_LENGTH = 16


#-------------------------------------------------------------------------

def inference(config: dict, save: bool):
    save_path = config["paths"]["expt_path"] / "predictions"
    save_path.mkdir(parents=True, exist_ok=True)
    patch_size = config["data"]["patch_size"]
    strides = [STRIDE_LENGTH, STRIDE_LENGTH, STRIDE_LENGTH]
    mb_size = config["expt"]["mb_size"]

    test_ds, TestGenerator = get_test_dataloader(
        config=config,
        by_subject=False,
        stride_length=STRIDE_LENGTH
    )

    model = build_model(config=config, purpose="inference")
    combine = CombinePatches(STRIDE_LENGTH)

    for data in test_ds:
        source = data["source"][0, ...]
        source = tf.transpose(source, [2, 0, 1])                   # TODO THIS NEEDS CHANGING
        subject_id = data["subject_id"][0].numpy().decode("utf-8")
        combine.new_subject(source.shape)

        linear_coords = generate_indices(source.shape, strides, patch_size)
        patch_stack = extract_patches(source, linear_coords, patch_size)
        patch_stack = tf.transpose(patch_stack, [0, 2, 3, 1, 4])    # TODO CHANGE
        num_patches = patch_stack.shape[0]
        pred_stack = []

        for i in range(0, num_patches, mb_size):
            pred_mb, _ = model(patch_stack[i:(i + mb_size), ...])
            pred_mb = TestGenerator.un_normalise(pred_mb)
            pred_stack.extend(pred_mb)

        pred_stack = tf.stack(pred_stack, axis=0)[:, :, :, :, 0]
        pred_stack = tf.transpose(pred_stack, [0, 3, 1, 2])    # TODO CHANGE
        combine.apply_patches(pred_stack, tf.stack(linear_coords, axis=0))
        pred = combine.get_img()
        pred = tf.transpose(pred, [1, 2, 0]).numpy()    # TODO CHANGE

        if save:
            img_nrrd = itk.GetImageFromArray(pred.astype("int16").transpose([2, 0, 1]))
            itk.WriteImage(img_nrrd, str(save_path / f"{subject_id}.nrrd"))
            print(f"{subject_id} saved")

        else:
            print(subject_id)
            plt.subplot(1, 2, 1)
            plt.imshow(pred[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(np.flipud(pred[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.show()


#-------------------------------------------------------------------------

def multiscale_inference(config: dict, save: bool):
    save_path = config["paths"]["expt_path"] / "predictions"
    save_path.mkdir(parents=True, exist_ok=True)
    patch_size = config["data"]["patch_size"]
    upscale_patch_size = (patch_size[2], patch_size[1] * 2, patch_size[0] * 2)
    strides = (STRIDE_LENGTH, STRIDE_LENGTH, STRIDE_LENGTH)
    upscale_strides = (STRIDE_LENGTH, STRIDE_LENGTH * 2, STRIDE_LENGTH * 2)
    mb_size = config["expt"]["mb_size"]
    dn_samp = config["hyperparameters"]["scales"][0]
    num_scales = len(config["hyperparameters"]["scales"])

    test_ds, TestGenerator = get_test_dataloader(
        config=config,
        by_subject=False,
        stride_length=STRIDE_LENGTH
    )

    model = build_model(config=config, purpose="inference")
    combine = CombinePatches(STRIDE_LENGTH)

    for data in test_ds:
        source = data["source"][0, ...]
        source = source[::dn_samp, ::dn_samp, :]
        source = tf.transpose(source, [2, 0, 1])                    # TODO CHANGE
        subject_id = data["subject_id"][0].numpy().decode("utf-8")

        linear_coords = generate_indices(source.shape, strides, patch_size)
        patch_stack = extract_patches(source, linear_coords, patch_size)
        patch_stack = tf.transpose(patch_stack, [0, 2, 3, 1, 4])    # TODO CHANGE
        num_patches = patch_stack.shape[0]
        pred_stack = []

        for i in range(0, num_patches, mb_size):
            pred_mb, _ = model(patch_stack[i:(i + mb_size), ...])
            pred_stack.extend(pred_mb)

        pred_stack = tf.stack(pred_stack, axis=0)[:, :, :, :, 0]
        pred_stack = tf.transpose(pred_stack, [0, 3, 1, 2])    # TODO CHANGE
        upscale_dims = (source.shape[0], source.shape[1] * 2, source.shape[2] * 2)
        combine.new_subject(upscale_dims)
        upscale_linear_coords = generate_indices(upscale_dims, upscale_strides, upscale_patch_size)
        combine.apply_patches(pred_stack, tf.stack(upscale_linear_coords, axis=0))
        pred = combine.get_img(False)
        pred = tf.transpose(pred, [1, 2, 0]).numpy()    # TODO CHANGE

        for _ in range(1, num_scales - 1):
            pred = tf.transpose(pred, [2, 0, 1])                    # TODO CHANGE
            upscale_dims = (upscale_dims[0], upscale_dims[1] * 2, upscale_dims[2] * 2)
            combine.new_subject(upscale_dims)
            linear_coords = generate_indices(pred.shape, strides, patch_size)
            patch_stack = extract_patches(pred, linear_coords, patch_size)
            patch_stack = tf.transpose(patch_stack, [0, 2, 3, 1, 4])    # TODO CHANGE
            num_patches = patch_stack.shape[0]
            pred_stack = []

            for i in range(0, num_patches, mb_size):
                pred_mb, _ = model(patch_stack[i:(i + mb_size), ...])
                pred_stack.extend(pred_mb)

            pred_stack = tf.stack(pred_stack, axis=0)[:, :, :, :, 0]
            pred_stack = tf.transpose(pred_stack, [0, 3, 1, 2])    # TODO CHANGE
            upscale_linear_coords = generate_indices(upscale_dims, upscale_strides, upscale_patch_size)
            combine.apply_patches(pred_stack, tf.stack(upscale_linear_coords, axis=0))
            pred = combine.get_img(False)
            pred = tf.transpose(pred, [1, 2, 0])    # TODO CHANGE

        pred = TestGenerator.un_normalise(pred)
        pred = tf.cast(tf.round(pred), "int16").numpy()

        if save:
            img_nrrd = itk.GetImageFromArray(pred.astype("int16").transpose([2, 0, 1]))
            itk.WriteImage(img_nrrd, str(save_path / f"{subject_id}.nrrd"))
            print(f"{subject_id} saved")

        else:
            print(subject_id)
            plt.subplot(1, 2, 1)
            plt.imshow(pred[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(np.flipud(pred[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.show()


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

    if len(config["hyperparameters"]["scales"]) == 1:
        inference(config, arguments.save)
    else:
        multiscale_inference(config, arguments.save)


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
