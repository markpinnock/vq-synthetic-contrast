import argparse
import glob
import matplotlib.pyplot as plt
import SimpleITK as itk
import numpy as np
import os
import tensorflow as tf
import yaml

from syntheticcontrast_v02.networks.models import get_model
from syntheticcontrast_v02.utils.build_dataloader import get_test_dataloader
from syntheticcontrast_v02.utils.combine_patches import CombinePatches


#-------------------------------------------------------------------------

def interpolate(CONFIG, args):
    times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
             2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0, 20.0]

    test_ds_dict, TestGenerator = get_test_dataloader(config=CONFIG,
                                                      by_subject=True,
                                                      mb_size=args.minibatch,
                                                      stride_length=args.stride)

    Model = get_model(config=CONFIG, purpose="inference")

    Combine = CombinePatches(CONFIG)

    for subject in list(test_ds_dict.keys()):
        subject_preds = []

        # Get original img dims
        original_img = glob.glob(f"{CONFIG['data']['data_path']}/Images/{subject}*")[0]
        img_dims = np.load(original_img).shape
        Combine.new_subject(img_dims)

        for time in times:
            for data in test_ds_dict[subject]:
                pred = Model(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * time)
                pred = TestGenerator.un_normalise(pred)[:, :, :, :, 0].numpy()

                Combine.apply_patches(pred, None, data["coords"])

            subject_preds.append(Combine.get_AC())
            Combine.reset()

        if args.save:
            for time, pred in zip(times, subject_preds):
                time_str = str(time).replace('.', '_')
                save_path = f"{CONFIG['paths']['expt_path']}/interpolation/{time_str}"
                print(f"{subject}, {time_str} saved")
                if not os.path.exists(save_path): os.makedirs(save_path)
                itk.WriteImage(itk.GetImageFromArray(pred.transpose([2, 0, 1])), f"{save_path}/{subject[0:6]}AP{subject[-3:]}.nrrd")

        else:
            print(subject)

            _, axs = plt.subplots(4, 4)

            for i in range(len(times)):
                axs.ravel()[i].imshow(subject_preds[i][:, :, 32], cmap="gray", vmin=-150, vmax=250)
                axs.ravel()[i].set_title(times[i])
                axs.ravel()[i].set_axis_off()

            plt.show()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Inference routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--minibatch", '-m', help="Minibatch size", type=int, default=128)
    parser.add_argument("--stride", '-st', help="Stride length", type=int, default=16)
    parser.add_argument("--save", '-s', help="Save images", action="store_true")
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path
    CONFIG["data"]["data_path"] = arguments.data
    
    interpolate(CONFIG, arguments)
