import json
import numpy as np
import pandas as pd
import plotly.express as px


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2"

with open(f"{FILE_PATH}/P2P_expts.json", 'r') as fp:
    results = json.load(fp)

img_dict = {"VPOOR": 0, "POOR": 1, "VBLURRY": 2, "BLURRY": 3}
df = pd.DataFrame()

for k in results:
    for l in results[k]:
        df.loc[k, l] = results[k][l]

df["images"] = df["images"].map(img_dict)
df[["d_eta", "g_eta", "lambda"]] = df[["d_eta", "g_eta", "lambda"]].apply(lambda x: np.log10(x))
# df.dropna(axis=0, inplace=True)
df.fillna(4, inplace=True)
df.drop("status", axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
fig = px.parallel_coordinates(
    df, color="images",
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2)

fig.show()