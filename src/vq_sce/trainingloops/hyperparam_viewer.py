import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def json_to_df(j):
    res = {
        "VPOOR": {"d_eta": [], "g_eta": [], "d_ch": [], "g_ch": [], "d_layers": [], "g_layers": [], "lambda": [], "mu": []},
        "POOR": {"d_eta": [], "g_eta": [], "d_ch": [], "g_ch": [], "d_layers": [], "g_layers": [], "lambda": [], "mu": []},
        "BLURRY": {"d_eta": [], "g_eta": [], "d_ch": [], "g_ch": [], "d_layers": [], "g_layers": [], "lambda": [], "mu": []},
        "DECENT": {"d_eta": [], "g_eta": [], "d_ch": [], "g_ch": [], "d_layers": [], "g_layers": [], "lambda": [], "mu": []}
        }

    for val in j.values():
        for k, v in val.items():
            try:
                res[val["images"]][k].append(v)
            except KeyError:
                continue

    dfs = [pd.DataFrame(res["VPOOR"]), pd.DataFrame(res["POOR"]), pd.DataFrame(res["BLURRY"]), pd.DataFrame(res["DECENT"])]

    for i, df in enumerate(dfs):
        df["images"] = i

    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    # df[["d_eta", "g_eta", "lambda"]] = df[["d_eta", "g_eta", "lambda"]].apply(lambda x: np.log10(x))

    return df

def reduce_dims(d):
    new_df = pd.DataFrame(index=d.index)
    new_df["eta"] = d["g_eta"] - d["d_eta"]
    new_df["ch"] = d["g_ch"] - d["d_ch"]
    new_df["layers"] = d["g_layers"] - d["d_layers"]
    new_df[["lambda", "mu", "images"]] = d[["lambda", "mu", "images"]]

    return new_df

if __name__ == "__main__":

    with open("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase3/P2PT.json", 'r') as fp:
        P2PT = json.load(fp)

    with open("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase3/HP2PT.json", 'r') as fp:
        HP2PT = json.load(fp)

    P2P = json_to_df(P2PT)
    P2PT = json_to_df(HP2PT)

    print(P2P.groupby("images").quantile([0.05, 0.5, 0.95]))
    print(P2PT.groupby("images").quantile([0.05, 0.5, 0.95]))
    # P2Pn = reduce_dims(P2P)
    # P2PTn = reduce_dims(P2PT)

    # A = pd.concat([P2P, P2PT], axis=0).reset_index(drop=True)
    # B = pd.concat([P2Pn, P2PTn], axis=0).reset_index(drop=True)

    # mean = A.groupby("images").mean()
    # upper = mean + 1.96 * A.groupby("images").std()
    # lower = mean - 1.96 * A.groupby("images").std()
    # print(mean)
    # print(upper)
    # print(lower)
    
    # plt.plot(mean["lambda"], 'k-')
    # plt.plot(upper["lambda"], 'k--')
    # plt.plot(lower["lambda"], 'k--')
    # plt.plot(mean["mu"], 'r-')
    # plt.plot(upper["mu"], 'r--')
    # plt.plot(lower["mu"], 'r--')
    # plt.show()