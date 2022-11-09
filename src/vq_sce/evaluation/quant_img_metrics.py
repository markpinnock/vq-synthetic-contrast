import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import pandas as pd
import skimage.metrics

from .util import bootstrap

np.set_printoptions(4)


#-------------------------------------------------------------------------

def calc_metrics(real_path, pred_path):
    real_path += "/Images"
    pred_path += "/Images"
    pred_imgs = os.listdir(pred_path)
    subjects = []
    MSE = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    pSNR = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    SSIM = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    
    for img in pred_imgs:
        if img[0:6] not in subjects:
            subjects.append(img[0:6])

    for subject in subjects:
        HQ, _ = nrrd.read(glob.glob(f"{real_path}/{subject}HQ*")[0])
        AC, _ = nrrd.read(glob.glob(f"{real_path}/{subject}AC*")[0])
        VC, _ = nrrd.read(glob.glob(f"{real_path}/{subject}VC*")[0])
        AP, _ = nrrd.read(glob.glob(f"{pred_path}/{subject}AP*")[0])
        VP, _ = nrrd.read(glob.glob(f"{pred_path}/{subject}VP*")[0])
    
        MSE["AP"].append(skimage.metrics.mean_squared_error(AC, AP))
        MSE["VP"].append(skimage.metrics.mean_squared_error(VC, VP))
        MSE["HQAC"].append(skimage.metrics.mean_squared_error(AC, HQ))
        MSE["HQVC"].append(skimage.metrics.mean_squared_error(VC, HQ))
        pSNR["AP"].append(skimage.metrics.peak_signal_noise_ratio(AC, AP))
        pSNR["VP"].append(skimage.metrics.peak_signal_noise_ratio(VC, VP))
        pSNR["HQAC"].append(skimage.metrics.peak_signal_noise_ratio(AC, HQ))
        pSNR["HQVC"].append(skimage.metrics.peak_signal_noise_ratio(AC, AP))
        SSIM["AP"].append(skimage.metrics.structural_similarity(AC, AP))
        SSIM["VP"].append(skimage.metrics.structural_similarity(VC, VP))
        SSIM["HQAC"].append(skimage.metrics.structural_similarity(AC, HQ))
        SSIM["HQVC"].append(skimage.metrics.structural_similarity(VC, HQ))

    return MSE, pSNR, SSIM


#-------------------------------------------------------------------------

def bootstrap_and_display(expt1, expt2, results):
    if expt2 is None:
        diff = np.median(results[expt1])
        boot_results = bootstrap(np.array(results[expt1]), None, N=100000)
    else:
        diff = np.median(results[expt1]) - np.median(results[expt2])
        boot_results = bootstrap(np.array(results[expt1]), np.array(results[expt2]), N=100000)

    h = plt.hist(boot_results, bins=20)
    plt.axvline(diff, c='k', ls='--')
    plt.errorbar(x=diff, y=(0.75 * np.max(h[0])), xerr=(1.96 * np.std(boot_results)))
    plt.title(f"{expt1} - {expt2}")
    plt.show()

    # Pivot method
    percentiles = np.quantile(boot_results, [0.975, 0.025]) # NB: these are switched

    return expt1, expt2, diff, 2 * diff - percentiles, f"Bias {np.mean(boot_results) - diff}, std err {np.std(boot_results)}"


#-------------------------------------------------------------------------

def save_to_csv(MSE, pSNR, SSIM, save_path):
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([["CME", "NGE"], ["MSE", "pSNR", "SSIM"]]))
    df["CME", "MSE"] = MSE["AP"]
    df["NGE", "MSE"] = MSE["VP"]
    df["CME", "pSNR"] = pSNR["AP"]
    df["NGE", "pSNR"] = pSNR["VP"]
    df["CME", "SSIM"] = SSIM["AP"]
    df["NGE", "SSIM"] = SSIM["VP"]
    df.to_csv(save_path)
    print(df)


#-------------------------------------------------------------------------

if __name__ == "__main__":

    models = {
        "UNetACVC": "imgqual_unetbase",
        "UNetT_save1000": "imgqual_unetphase",
        "CycleGANT_save880": "imgqual_cyclegan",
        "2_save230": "imgqual_p2p",
        "2_save170_patch": "imgqual_p2ppatch",
        "H2_save280": "imgqual_hyperp2p",
        "H2_save300_patch": "imgqual_hyperp2ppatch"
    }

    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"

    for model, save_name in models.items():
        print(model)
        pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}"
        save_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/results/{save_name}.csv"
        MSE, pSNR, SSIM = calc_metrics(real_path, pred_path)
        save_to_csv(MSE, pSNR, SSIM, save_path)

    exit()
    print(bootstrap_and_display("AP", None, MSE))
    print(bootstrap_and_display("VP", None, MSE))
    print(bootstrap_and_display("AP", None, pSNR))
    print(bootstrap_and_display("VP", None, pSNR))
    print(bootstrap_and_display("AP", None, SSIM))
    print(bootstrap_and_display("VP", None, SSIM))
    