import json
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------------------

def display_subjects(HUs: str):

    for subject in HUs.keys():
        Ao = np.array(HUs[subject]["Ao"])
        RK = np.array(HUs[subject]["RK"])
        LK = np.array(HUs[subject]["LK"])
        Tu = np.array(HUs[subject]["Tu"])
        t = np.array(HUs[subject]["times"])
        t[t < 0.0] = 0

        plt.subplot(2, 2, 1)
        plt.plot(t, Ao, label="Ao")
        plt.subplot(2, 2, 2)
        plt.plot(t, RK, label="RK")
        plt.subplot(2, 2, 3)
        plt.plot(t, LK, label="LK")
        plt.subplot(2, 2, 4)
        plt.plot(t, Tu, label="Tu")
        plt.xlabel("Series")
        plt.ylabel("HU")
        plt.title(subject)
        plt.legend()
        plt.show()


#-------------------------------------------------------------------------

def interpolate(HUs):
    grid = np.arange(400, 11000, 500)

    HU_0 = {
        "Ao": np.zeros(len(HUs.keys())),
        "RK": np.zeros(len(HUs.keys())),
        "LK": np.zeros(len(HUs.keys())),
        "Tu": np.zeros(len(HUs.keys()))
    }
    HU_35 = {
        "Ao": np.zeros(len(HUs.keys())),
        "RK": np.zeros(len(HUs.keys())),
        "LK": np.zeros(len(HUs.keys())),
        "Tu": np.zeros(len(HUs.keys()))
    }
    HU_90 = {
        "Ao": np.zeros(len(HUs.keys())),
        "RK": np.zeros(len(HUs.keys())),
        "LK": np.zeros(len(HUs.keys())),
        "Tu": np.zeros(len(HUs.keys()))
    }
    delayed = {
        "Ao": np.zeros((len(HUs.keys()), grid.shape[0])),
        "RK": np.zeros((len(HUs.keys()), grid.shape[0])),
        "LK": np.zeros((len(HUs.keys()), grid.shape[0])),
        "Tu": np.zeros((len(HUs.keys()), grid.shape[0]))
    }

    for i, subject in enumerate(HUs.keys()):
        t = np.array(HUs[subject]["times"])[3:]

        for key in delayed.keys():
            HU_0[key][i] = HUs[subject][key][0]
            HU_35[key][i] = HUs[subject][key][1]
            HU_90[key][i] = HUs[subject][key][2]
        
            idx = np.digitize(t, grid)
            delayed[key][i, idx] = HUs[subject][key][3:]

    return {"0": HU_0, "35": HU_35, "90": HU_90, "delayed": delayed}


#-------------------------------------------------------------------------

def display_aggregate(HUs, num_to_include=None):
    grid = np.arange(400, 11000, 500)
    HU_0 = HUs["0"]
    HU_35 = HUs["35"]
    HU_90 = HUs["90"]
    delayed = HUs["delayed"]

    mean_d = dict.fromkeys(delayed.keys())
    std_d = dict.fromkeys(delayed.keys())

    times_to_include = (delayed["Ao"] != 0).sum(axis=0) > 0
    times = np.hstack([0.0, 35.0, 90.0, grid[times_to_include]])[0:num_to_include]

    for key in delayed.keys():
        delayed[key] = delayed[key][:, times_to_include]
        mean_d[key] = np.hstack([HU_0[key].mean(), HU_35[key].mean(), HU_90[key].mean(), delayed[key].mean(axis=0)])[0:num_to_include]
        std_d[key] = np.hstack([HU_0[key].std(), HU_35[key].std(), HU_90[key].std(), delayed[key].std(axis=0)])[0:num_to_include]

    plt.subplot(2, 2, 1)
    plt.errorbar(times / 60, mean_d["Ao"], 1.96 * std_d["Ao"], ecolor='r')
    plt.xlabel("Time (min)")
    plt.ylabel("HU")
    plt.title("Aorta")
    plt.subplot(2, 2, 2)
    plt.errorbar(times / 60, mean_d["RK"], 1.96 * std_d["RK"], ecolor='r')
    plt.xlabel("Time (min)")
    plt.ylabel("HU")
    plt.title("Right Kidney")
    plt.subplot(2, 2, 3)
    plt.errorbar(times / 60, mean_d["LK"], 1.96 * std_d["LK"], ecolor='r')
    plt.xlabel("Time (min)")
    plt.ylabel("HU")
    plt.title("Left Kidney")
    plt.subplot(2, 2, 4)
    plt.errorbar(times / 60, mean_d["Tu"], 1.96 * std_d["Tu"], ecolor='r')
    plt.xlabel("Time (min)")
    plt.ylabel("HU")
    plt.title("Tumour")
    plt.show()

#-------------------------------------------------------------------------

def one_compartment(HUs):
    grid = np.arange(400, 11000, 500)
    HU_0 = HUs["0"]
    HU_35 = HUs["35"]
    HU_90 = HUs["90"]
    delayed = HUs["delayed"]

    mean_d = dict.fromkeys(delayed.keys())
    std_d = dict.fromkeys(delayed.keys())

    times_to_include = (delayed["Ao"] != 0).sum(axis=0) > 0
    times = np.hstack([35.0, 90.0, grid[times_to_include]])

    for key in delayed.keys():
        delayed[key] = delayed[key][:, times_to_include]
        mean_d[key] = np.hstack([HU_35[key].mean(), HU_90[key].mean(), delayed[key].mean(axis=0)])
        std_d[key] = np.hstack([HU_35[key].std(), HU_90[key].std(), delayed[key].std(axis=0)])

    y = np.log(mean_d["Ao"])[:, np.newaxis]
    X = np.hstack([np.ones_like(y), times[:, np.newaxis]])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(np.exp(beta[0]), 1 / beta[1] / 60, 1 / beta[1] * np.log(2) / 60)

    y = np.log(mean_d["RK"])[:, np.newaxis]
    X = np.hstack([np.ones_like(y), times[:, np.newaxis]])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(np.exp(beta[0]), 1 / beta[1] / 60, 1 / beta[1] * np.log(2) / 60)

    y = np.log(mean_d["LK"])[:, np.newaxis]
    X = np.hstack([np.ones_like(y), times[:, np.newaxis]])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(np.exp(beta[0]), 1 / beta[1] / 60, 1 / beta[1] * np.log(2) / 60)

    y_hat = np.exp(beta[0]) * np.exp(beta[1] * np.hstack([35.0, 90.0, grid[times_to_include]]))

    plt.scatter(times, mean_d["Ao"])
    plt.plot(np.hstack([35.0, 90.0, grid[times_to_include]]), y_hat, 'k')
    plt.show()

#-------------------------------------------------------------------------

if __name__ == "__main__":
    with open("syntheticcontrast_v02/contrastmodelling/HUs.json", 'r') as fp:
        HU = json.load(fp)

    display_subjects(HU)
    HU_agg = interpolate(HU)
    # display_aggregate(HU_agg)
    one_compartment(HU_agg)
