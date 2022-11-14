import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import SimpleITK as itk
import tensorflow as tf
import yaml

from vq_sce.networks.build_model import build_model
from vq_sce.utils.build_dataloader import get_test_dataloader
from vq_sce.utils.combine_patches import CombinePatches
from vq_sce.utils.patch_utils import extract_patches

STRIDE_LENGTH = 16


#-------------------------------------------------------------------------

def inference(config: dict, save: bool):
    patch_size = config["data"]["patch_size"]
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
        subject_id = data["subject_id"]
        combine.new_subject(source.shape)

        patches, indices = extract_patches(source, STRIDE_LENGTH, patch_size)
        patch_stack = tf.stack(patches, axis=0)
        num_patches = patch_stack.shape[0]
        pred_list = []

        for i in range(0, num_patches, mb_size):
            print(i, i + mb_size)
            pred_mb = model(patch_stack[i:(i + mb_size), ...])
            pred_list.append(pred_mb)

        pred_stack = tf.stack(pred_list, axis=0)
        pred_stack = TestGenerator.un_normalise(pred_stack)[:, :, :, :, 0]

        combine.apply_patches(pred_stack, indices)
        pred = combine.get_img()

        if save:
            save_path = config["paths"]["expt_path"] / "predictions"
            save_path.mkdir(parents=True, exist_ok=True)
            img_nrrd = itk.GetImageFromArray(pred).astype("int16").transpose([2, 0, 1])
            itk.WriteImage(img_nrrd, save_path / f"{subject_id}.nrrd")
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
    
    inference(config, arguments.save)


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
