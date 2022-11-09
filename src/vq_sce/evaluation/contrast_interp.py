import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import pandas as pd

np.set_printoptions(4)


#-------------------------------------------------------------------------

def calc_contrast(real_path, pred_path, save_path, model_save_name=None):

    HQ = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    CE = {}
    AC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    VC = {"Ao": [], "Co": [], "Md": [], "Tu": []}

    times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
             2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0, 20.0]

    flag = True

    times_dict = dict(zip([str(t).replace('.', '_') for t in times], times))

    for time in times_dict.keys():
        CE[time] = {"Ao": [], "Co": [], "Md": [], "Tu": []}
        preds = os.listdir(f"{pred_path}/{time}")

        for pred in preds:
            # Get predicted img
            img, _ = nrrd.read(f"{pred_path}/{time}/{pred}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

            # Average slices for masks
            CE[time]["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
            CE[time]["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
            CE[time]["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            # Average whole tumour mask
            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
            except:
                CE[time]["Tu"].append(np.nan)
            else:
                CE[time]["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            img, _ = nrrd.read(f"{real_path}/Images/{pred[0:6]}HQ{pred[8:11]}.nrrd")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

            if flag:
                # Get ground truth HQ phase
                # Average slices for masks
                HQ["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                HQ["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                HQ["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
                except:
                    HQ["Tu"].append(np.nan)
                else:
                    HQ["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

                candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}AC*.nrrd")
                assert len(candidate_imgs) == 1
                img_name = candidate_imgs[0].split('\\')[-1]
                img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
                seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

                # Get ground truth AC phase
                # Average slices for masks
                AC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                AC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                AC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
                except:
                    AC["Tu"].append(np.nan)
                else:
                    AC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

                candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}VC*.nrrd")
                assert len(candidate_imgs) == 1
                img_name = candidate_imgs[0].split('\\')[-1]

                img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
                seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

                # Get ground truth VC phase
                # Average slices for masks
                VC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                VC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                VC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
                except:
                    VC["Tu"].append(np.nan)
                else:
                    VC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())
    
        flag = False

    if model_save_name is not None:
        gt_cols = pd.MultiIndex.from_product([["NCE", "CME", "NGE"], ["Aorta", "Cortex", "Medulla", "Tumour"]])
        pred_cols = pd.MultiIndex.from_product([times_dict.keys(), ["Aorta", "Cortex", "Medulla", "Tumour"]])
        gt_df = pd.DataFrame(columns=gt_cols)
        pred_df = pd.DataFrame(columns=pred_cols)

        for ROI_old, ROI_new in zip(["Ao", "Co", "Md", "Tu"], ["Aorta", "Cortex", "Medulla", "Tumour"]):
            gt_df["NCE", ROI_new] = HQ[ROI_old]
            gt_df["CME", ROI_new] = AC[ROI_old]
            gt_df["NGE", ROI_new] = VC[ROI_old]

        for ROI_old, ROI_new in ROI_dict.items():
            for time in times_dict.keys():
                pred_df[time, ROI_new] = CE[time][ROI_old]
                pred_df[time, ROI_new] = CE[time][ROI_old]

        gt_df.to_csv(f"{save_path}/interp_gt.csv")
        pred_df.to_csv(f"{save_path}/interp_{model_save_name}.csv")


#-------------------------------------------------------------------------

def load_contrasts(file_path, models):
    results = []

    times = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75",
             "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "10.0", "20.0"]

    for model in models.keys():
        model_dict = {}

        for time in times:
            model_dict[time] = {}

        df = pd.read_csv(f"{file_path}/interp_{model}.csv", index_col=0, header=[0, 1])

        for ROI in ["Aorta", "Cortex", "Medulla", "Tumour"]:
            for time in times:
                model_dict[time][ROI] = df[time.replace('.', '_'), ROI].dropna().values

        results.append(model_dict)

    return dict(zip(models.values(), results))


#-------------------------------------------------------------------------

def display_interps(file_path, results):
    gt_df = pd.read_csv(f"{file_path}/contrast_gt.csv", index_col=0, header=[0, 1])

    for i, ROI in enumerate(["Aorta", "Cortex", "Medulla", "Tumour"]):
        plt.subplot(2, 2, i + 1)

        for model in models.values():
            times = list(results[model].keys())
            times_float = [float(t) for t in results[model].keys()]
            HUs = []

            for t in times:
                HUs.append(np.mean(results[model][t][ROI]))

            plt.plot(times_float, HUs, label=model)

        plt.scatter(0.0, gt_df.mean(axis=0)["NCE"][ROI])
        plt.scatter(1.0, gt_df.mean(axis=0)["CME"][ROI])
        plt.scatter(2.0, gt_df.mean(axis=0)["NGE"][ROI])

    plt.legend()
    plt.show()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    models = {
        "p2p": "P2P-Full",
        "p2ppatch": "P2P-Patch",
        "hp2p": "HyperP2P-Full",
        "hp2ppatch": "HyperP2P-Patch"
    }

    s = "C:\\Users\\roybo\\OneDrive - University College London\\PhD\\PhD_Prog\\007_CNN_Virtual_Contrast\\Phase2\\results"

    results = load_contrasts(s, models)
    # display_interps(s, results)
    # p = "C:\\Users\\roybo\\Programming\\PhD\\007_CNN_Virtual_Contrast\\test_pix2pix\\H2_save300_patch\\interpolation"
    # r = "C:\\Users\\roybo\\OneDrive - University College London\\PhD\\PhD_Prog\\007_CNN_Virtual_Contrast\\Phase2\\output\\Real"
    # calc_contrast(r, p, s, "hp2ppatch")

    segs = pd.read_csv("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/results/seg_interp.csv", index_col=0)

    times_f = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
               2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0, 20.0]

    # plt.plot(times_f, segs.median(axis=0)[0:16], label="P2P-Full")
    # plt.legend()
    # plt.show()
    # plt.plot(times_f, segs.median(axis=0)[16:32], label="P2P-Patch")
    # plt.legend()
    # plt.show()
    # plt.plot(times_f, segs.median(axis=0)[32:48], label="HyperP2P-Full")
    # plt.legend()
    # plt.show()
    # plt.plot(times_f, segs.median(axis=0)[48:64], label="HyperP2P-Patch")
    # plt.legend()
    # plt.show()

    print(segs.median(axis=0)[0:16])
    print(segs.median(axis=0)[16:32])
    print(segs.median(axis=0)[32:48])
    print(segs.median(axis=0)[48:])
