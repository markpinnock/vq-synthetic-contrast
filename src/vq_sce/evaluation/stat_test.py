import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stat

from statsmodels.formula.api import ols

from .contrast_test import calc_contrast
from .quant_img_metrics import calc_metrics


#-------------------------------------------------------------------------

def get_contrast_results(real_path, pred_path, models):
    contrasts = {}

    for model in models:
        contrasts[model] = calc_contrast(real_path, f"{pred_path}/{model}")

    return contrasts


#-------------------------------------------------------------------------

def get_metric_results(real_path, pred_path, models):
    MSE = {}
    pSNR = {}
    SSIM = {}

    for model in models:
        print(model)
        M, p, S = calc_metrics(real_path, f"{pred_path}/{model}")
        MSE[model] = M
        pSNR[model] = p
        SSIM[model] = S

    return MSE, pSNR, SSIM


#-------------------------------------------------------------------------

def get_mannwhitneyu(expt1, expt2, phase, c):
    e1 = c[expt1][phase]
    e2 = c[expt2][phase]
    print(expt1, expt2, phase, stat.mannwhitneyu(e1, e2))


#-------------------------------------------------------------------------

def check_interactions(contrast_dict, gt_phase, formula):
    pred_phase = f"{gt_phase[0]}P"

    for ROI in ["Ao", "Co", "Md", "Tu"]:
        y = []
        x = []
        g = []

        for i, expt in enumerate(contrast_dict.keys()):
            print(i, expt)
            y.append(np.array(contrast_dict[expt][gt_phase][ROI]) - np.array(contrast_dict[expt][pred_phase][ROI]))
            x.append((np.array(contrast_dict[expt][gt_phase][ROI]) + np.array(contrast_dict[expt][pred_phase][ROI])) / 2)
            g.append(np.ones_like(x[-1]) * i)

        df = pd.DataFrame({'y': np.hstack(y), 'x': np.hstack(x) - np.hstack(x).mean(), 'g': np.hstack(g)})

        lm = ols(formula, df).fit()
        print(ROI, np.hstack(x).mean(), gt_phase)
        print("=========================================================================================")
        print(lm.summary())


#-------------------------------------------------------------------------

def check_contrasts(contrast_dict, gt_phase, formula, contrast):
    pred_phase = f"{gt_phase[0]}P"

    for ROI in ["Ao", "Co", "Md", "Tu"]:
        y = []
        x = []
        g = []

        for i, expt in enumerate(contrast_dict.keys()):
            print(i, expt)
            y.append(np.array(contrast_dict[expt][gt_phase][ROI]) - np.array(contrast_dict[expt][pred_phase][ROI]))
            x.append((np.array(contrast_dict[expt][gt_phase][ROI]) + np.array(contrast_dict[expt][pred_phase][ROI])) / 2)
            g.append(np.ones_like(x[-1]) * i)

        df = pd.DataFrame({'y': np.hstack(y), 'x': np.hstack(x) - np.nanmean(np.hstack(x)), 'g': np.hstack(g)})
        df.dropna(axis=0, inplace=True)

        lm = ols(formula, df).fit()
        print(ROI, np.hstack(x).mean(), gt_phase)
        print("=========================================================================================")
        print(lm.summary())
        print("\n")

        if contrast == "intercept":
            lambda_ = np.array([[0], [1], [0], [0], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            
            lambda_ = np.array([[0], [0], [1], [0], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            
            lambda_ = np.array([[0], [0], [0], [1], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            
            lambda_ = np.array([[0], [-1], [1], [0], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            
            lambda_ = np.array([[0], [-1], [0], [1], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            
            lambda_ = np.array([[0], [0], [-1], [1], [0], [0], [0], [0]])
            t = np.squeeze((lambda_.T @ lm.params.values) / np.sqrt(lambda_.T @ lm.cov_params().values @ lambda_))
            print(np.squeeze(lambda_), t, (1 - stat.t.cdf(np.abs(t), df.shape[0])) * 2)
            

        elif contrast == "slope":
            pass

        print("\n")


#-------------------------------------------------------------------------

def test_img_qual_metrics(real_path, pred_path, models):
    MSE, pSNR, SSIM = get_metric_results(real_path, pred_path, models)
    model_idx = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    print("MSE")
    to_test = []

    for model in models:
        to_test.append(MSE[model]["AP"])

    H, p = stat.kruskal(*to_test)
    print(f"AP, Kruskall H {H}, p {p}")

    to_test = []

    for model in models:
        to_test.append(MSE[model]["VP"])

    H, p = stat.kruskal(*to_test)
    print(f"VP, Kruskall H {H}, p {p}")

    for idx in model_idx:
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "AP", MSE)
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "VP", MSE)

    print("pSNR")
    to_test = []

    for model in models:
        to_test.append(pSNR[model]["AP"])

    H, p = stat.kruskal(*to_test)
    print(f"AP, Kruskall H {H}, p {p}")

    to_test = []

    for model in models:
        to_test.append(pSNR[model]["VP"])

    H, p = stat.kruskal(*to_test)
    print(f"VP, Kruskall H {H}, p {p}")

    for idx in model_idx:
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "AP", pSNR)
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "VP", pSNR)

    print("SSIM")
    to_test = []

    for model in models:
        to_test.append(SSIM[model]["AP"])

    H, p = stat.kruskal(*to_test)
    print(f"AP, Kruskall H {H}, p {p}")

    to_test = []

    for model in models:
        to_test.append(SSIM[model]["VP"])

    H, p = stat.kruskal(*to_test)
    print(f"VP, Kruskall H {H}, p {p}")

    for idx in model_idx:
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "AP", SSIM)
        get_mannwhitneyu(models[idx[0]], models[idx[1]], "VP", SSIM)


#-------------------------------------------------------------------------

def test_contrasts(real_path, pred_path, models, type):
    contrast = get_contrast_results(real_path, pred_path, models)
    check_contrasts(contrast, "AC", "y ~ x * C(g)", type)
    print("==============================================================================================================")
    print("==============================================================================================================")
    check_contrasts(contrast, "VC", "y ~ x * C(g)", type)


#-------------------------------------------------------------------------

if __name__ == "__main__":
    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    pred_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output"
    models = ["2_save230", "2_save170_patch", "H2_save280", "H2_save300_patch"]

    #test_img_qual_metrics(real_path, pred_path, models)
    test_contrasts(real_path, pred_path, models, "intercept")
