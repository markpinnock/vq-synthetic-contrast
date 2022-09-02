import json
import matplotlib.pyplot as plt
import numpy as np
import os

from .util import load_images, resample, display_imgs, get_HUs, aggregate_HUs


#-------------------------------------------------------------------------

def display_subjects(
    subject_list: list,
    subject_ignore: list = [],
    image_ignore: list = [],
    depth_idx: int = None,
    img_path: str = None,
    seg_path: str = None,
    trans_path: str = None,
    time_path: str = None
    ):
    for subject in subject_list:
        if subject in subject_ignore or 'F' in subject:
            continue

        print(subject)
        imgs, segs = load_images(subject, img_path, seg_path, trans_path, ignore=image_ignore)
        imgs, segs = resample(imgs, segs)

        AC = [n for n in imgs.keys() if 'AC' in n]
        VC = [n for n in imgs.keys() if 'VC' in n]
        HQ = [n for n in imgs.keys() if 'HQ' in n]
        keys = sorted(AC + VC + HQ, key=lambda x: int(x[-3:]))

        with open(f"{time_path}/{subject}/time.json", 'r') as fp:
            times = json.load(fp)

        t = [times[f"{k}.nrrd"] for k in keys]
        overlay = np.copy(segs[AC[0]])

        display_imgs(imgs, segs, keys, overlay=overlay, depth_idx=depth_idx)
        display_HUs(imgs, segs[AC[0]], keys, t)


#-------------------------------------------------------------------------

def display_HUs(imgs: dict, seg: object, keys: list, t: list = None):
    Ao, RK, LK, Tu = get_HUs(imgs, seg, keys)

    plt.figure(figsize=(18, 10))

    if t is None:
        plt.plot(Ao, label="Ao")
        plt.plot(RK, label="RK")
        plt.plot(LK, label="LK")
        plt.plot(Tu, label="Tu")

    else:
        plt.plot(t, Ao, label="Ao")
        plt.plot(t, RK, label="RK")
        plt.plot(t, LK, label="LK")
        plt.plot(t, Tu, label="Tu")

    plt.xlabel("Series")
    plt.ylabel("HU")
    plt.title(keys[0][0:6])
    plt.legend()
    plt.show()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    with open("syntheticcontrast_v02/preproc/ignore.json", 'r') as fp:
        ignore = json.load(fp)

    subject_ignore = list(ignore["subject_ignore"].keys())
    image_ignore = ignore["image_ignore"]

    IMG_PATH = "Z:/Clean_CT_Data/Toshiba/Images"
    SEG_PATH = "Z:/Clean_CT_Data/Toshiba/Segmentations"
    TRANS_PATH = "Z:/Clean_CT_Data/Toshiba/Transforms"
    TIME_PATH = "Z:/Clean_CT_Data/Toshiba/Times"
    subjects = os.listdir(IMG_PATH)

    subject_ignore += ["T016A0"]

    display_subjects(subjects, subject_ignore=subject_ignore, image_ignore=image_ignore, depth_idx=None, img_path=IMG_PATH, seg_path=SEG_PATH, trans_path=TRANS_PATH, time_path=TIME_PATH)
    # HUs = aggregate_HUs(subjects, subject_ignore=subject_ignore, image_ignore=image_ignore, times=None, img_path=IMG_PATH, seg_path=SEG_PATH, trans_path=TRANS_PATH, time_path=TIME_PATH)

    with open("C:/Users/roybo/Programming/PhD/007_CNN_Virtual_Contrast/syntheticcontrast_v02/contrastmodelling/HUs.json", 'w') as fp:
        json.dump(HUs, fp, indent=4)

# Not enough segs? T016A0
# No initial HQ? T042A1
