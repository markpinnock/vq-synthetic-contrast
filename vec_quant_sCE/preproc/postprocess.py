import argparse
import glob
import numpy as np
import os
import SimpleITK as itk


#-------------------------------------------------------------------------

def NRRDConv_v01(image_path, pred_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preds = [name for name in os.listdir(pred_path)]

    for pred_name in preds:
        HQ_name = f"{pred_name[0:6]}HQ{pred_name[8:]}"

        if "AP" in pred_name:
            CE_candidates = glob.glob(f"{image_path}/{pred_name[0:6]}AC*")
        elif "VP" in pred_name:
            CE_candidates = glob.glob(f"{image_path}/{pred_name[0:6]}VC*")
        else:
            raise ValueError

        assert len(CE_candidates) == 1, CE_candidates

        pred_nrrd = itk.GetImageFromArray(np.load(f"{pred_path}/{pred_name}").astype("int16").transpose([2, 0, 1]))
        HQ_nrrd = itk.GetImageFromArray(np.load(f"{image_path}/{HQ_name}").astype("int16").transpose([2, 0, 1]))
        CE_nrrd = itk.GetImageFromArray(np.load(f"{CE_candidates[0]}").astype("int16").transpose([2, 0, 1]))

        itk.WriteImage(pred_nrrd, f"{save_path}/{pred_name[:-4]}.nrrd")
        itk.WriteImage(HQ_nrrd, f"{save_path}/{HQ_name[:-4]}.nrrd")
        itk.WriteImage(CE_nrrd, f"{save_path}/{CE_candidates[0][-15:][:-4]}.nrrd")


#-------------------------------------------------------------------------

def NRRDConv_v02(image_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgs = [name for name in os.listdir(image_path)]

    for name in imgs:
        img_nrrd = itk.GetImageFromArray(np.load(f"{image_path}/{name}").astype("int16").transpose([2, 0, 1]))

        itk.WriteImage(img_nrrd, f"{save_path}/{name[:-4]}.nrrd")


#-------------------------------------------------------------------------

if __name__ == "__main__":

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--save", "-s", help="Save path", type=str)
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path
    NRRDConv_v02(arguments.path, arguments.save)
