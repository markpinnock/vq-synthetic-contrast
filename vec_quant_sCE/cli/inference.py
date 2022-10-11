import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import yaml

from syntheticcontrast_v02.networks.models import get_model
from syntheticcontrast_v02.utils.build_dataloader import get_test_dataloader
from syntheticcontrast_v02.utils.combine_patches import CombinePatches


#-------------------------------------------------------------------------

def inference(CONFIG, args):
    assert args.phase in ["AC", "VC", "both"], args.phase
    times = args.time.split(",")
    assert len(times) == 2
    times = [float(t) for t in times]

    test_ds_dict, TestGenerator = get_test_dataloader(config=CONFIG,
                                                      by_subject=True,
                                                      mb_size=args.minibatch,
                                                      stride_length=args.stride)

    Model = get_model(config=CONFIG, purpose="inference")

    Combine = CombinePatches(CONFIG)

    for subject, test_ds in test_ds_dict.items():

        # Get original img dims
        original_img = glob.glob(f"{CONFIG['data']['data_path']}/Images/{subject}*")[0]
        img_dims = np.load(original_img).shape
        Combine.new_subject(img_dims)

        for data in test_ds:
            AC_pred = Model(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * times[0])
            VC_pred = Model(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * times[1])

            AC_pred = TestGenerator.un_normalise(AC_pred)[:, :, :, :, 0].numpy()
            VC_pred = TestGenerator.un_normalise(VC_pred)[:, :, :, :, 0].numpy()

            Combine.apply_patches(AC_pred, VC_pred, data["coords"])

        AC = Combine.get_AC()
        VC = Combine.get_VC()

        if args.save:
            save_path = f"{CONFIG['paths']['expt_path']}/predictions"
            print(f"{subject} saved")
            if not os.path.exists(save_path): os.mkdir(save_path)

            if args.phase == "AC":
                np.save(f"{save_path}/{subject[0:6]}AP{subject[-3:]}", AC)
            elif args.phase == "VC":
                np.save(f"{save_path}/{subject[0:6]}VP{subject[-3:]}", VC)
            elif args.phase == "both":
                np.save(f"{save_path}/{subject[0:6]}AP{subject[-3:]}", AC)
                np.save(f"{save_path}/{subject[0:6]}VP{subject[-3:]}", VC)
            else:
                raise ValueError

        else:
            print(subject)
            plt.subplot(2, 2, 1)
            plt.imshow(AC[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(np.flipud(AC[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(VC[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(np.flipud(VC[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.show()
    

#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Inference routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--phase", '-f', help="Phase: AC/VC/both", type=str, default="both")
    parser.add_argument("--minibatch", '-m', help="Minibatch size", type=int, default=128)
    parser.add_argument("--stride", '-st', help="Stride length", type=int, default=16)
    parser.add_argument("--time", '-tt', help="Phase time, comma separated", type=str, default="1,2")
    parser.add_argument("--save", '-s', help="Save images", action="store_true")
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path
    CONFIG["data"]["data_path"] = arguments.data
    
    inference(CONFIG, arguments)
