import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from .unet import UNet
from .util import load_data, dice_loss, bootstrap

np.set_printoptions(4)

#-------------------------------------------------------------------------

def test(expt_path, time=None):

    with open(f"{expt_path}/config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    if time is not None:
        time_float = float(time)
        config["data_config"]["img_type"] = time_float

    # test_ds = tf.data.Dataset.from_tensor_slices(
    #     )
    test_ds = load_data("test", config["model_config"]["img_dims"][0], **config["data_config"])
    # segs = next(iter(test_ds.batch(256)))[1]
    imgs = np.stack(test_ds[0], axis=0)
    segs = np.stack(test_ds[1], axis=0)
    #test_ds = test_ds.batch(config["model_config"]["mb_size"])

    Model = UNet(config["model_config"])

    if time is not None:
        Model.load_weights(f"{expt_path}/{time.replace('.', '_')}/models/model.ckpt").expect_partial()
    else:
        Model.load_weights(f"{expt_path}/models/model.ckpt").expect_partial()

    prediction = Model(imgs).numpy()

    dice = 1 - dice_loss(prediction, segs, axis=[1, 2, 3])

    return dice


#-------------------------------------------------------------------------

def bootstrap_and_display(expt1, expt2, display=True):
    if expt2 is not None:
        diff = np.median(results[expt1]) - np.median(results[expt2])
        boot_results = bootstrap(results[expt1], results[expt2], N=100000)
    else:
        diff = np.median(results[expt1])
        boot_results = bootstrap(results[expt1], None, N=100000)

    if display:
        h = plt.hist(boot_results, bins=20)
        plt.axvline(diff, c='k', ls='--')
        plt.errorbar(x=diff, y=(0.75 * np.max(h[0])), xerr=(1.96 * np.std(boot_results)))
        plt.title(f"{expt1} - {expt2}")
        plt.show()

    # Pivot method
    percentiles = np.quantile(boot_results, [0.975, 0.025]) # NB: these are switched

    return expt1, expt2, diff, 2 * diff - percentiles, f"Bias {np.mean(boot_results) - diff}, std err {np.std(boot_results)}"


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Segmentation-based evaluation testing routine """

    expts_path = "syntheticcontrast_v02/evaluation/expts"
    expts = ["2_save230", "2_save170_patch", "H2_save280", "H2_save300_patch"]
    times = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75",
             "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "10.0", "20.0"]

    results = {}

    for expt in expts:
        for time in times:
            results[f"{expt}_{time}"] = test(f"{expts_path}/{expt}", time)

    df_interp = pd.DataFrame()

    for k, v in results.items():
        print(k, np.median(v), np.quantile(v, [0.05, 0.95]))
        df_interp[k] = v

    df_interp.to_csv("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/results/seg_interp.csv")
    exit()
    # plt.boxplot(box_results)
    # plt.xticks(list(range(1, len(expts) + 1)), expts)
    # plt.ylabel("Dice")
    # plt.show()

    print(bootstrap_and_display("AC", None, False))
    print(bootstrap_and_display("VC", None, False))
    print(bootstrap_and_display("HQ", None, False))
    print(bootstrap_and_display("UNetACVC_AP", None, False))
    print(bootstrap_and_display("UNetACVC_VP", None, False))
    print(bootstrap_and_display("UNetT_save1000_AP", None, False))
    print(bootstrap_and_display("UNetT_save1000_VP", None, False))
    print(bootstrap_and_display("CycleGANT_save880_AP", None, False))
    print(bootstrap_and_display("CycleGANT_save880_VP", None, False))
    print(bootstrap_and_display("2_save230_AP", None, False))
    print(bootstrap_and_display("2_save230_VP", None, False))
    print(bootstrap_and_display("2_save170_patch_AP", None, False))
    print(bootstrap_and_display("2_save170_patch_VP", None, False))
    print(bootstrap_and_display("H2_save280_AP", None, False))
    print(bootstrap_and_display("H2_save280_VP", None, False))
    print(bootstrap_and_display("H2_save300_patch_AP", None, False))
    print(bootstrap_and_display("H2_save300_patch_VP", None, False))
