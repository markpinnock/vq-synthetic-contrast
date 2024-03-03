import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as itk  # noqa: N813
from numpy.typing import NDArray

from vq_sce import MIN_HQ_DEPTH
from vq_sce.networks.model import Task

# -------------------------------------------------------------------------


def trim_lq(source: itk.Image) -> tuple[itk.Image, int, int]:
    source_slice_means = itk.GetArrayFromImage(source).mean(axis=(1, 2))
    source_slice_idx = np.argwhere(source_slice_means > -1)
    source_lower = int(source_slice_idx[0])
    source_upper = int(source_slice_idx[-1]) + 1

    if (source_upper - source_lower) != MIN_HQ_DEPTH:
        raise ValueError(source_lower, source_upper)

    return source, source_lower, source_upper


# -------------------------------------------------------------------------


def calc_dice(source: NDArray[np.int16], target: NDArray[np.int16]) -> float:
    numerator = 2 * (source * target).sum()
    denominator = source.sum() + target.sum() + 1e-12

    return numerator / denominator


# -------------------------------------------------------------------------


def load_and_transform(
    lq_id: str,
    paths: dict[str, Path],
    ac_or_hq: str,
) -> tuple[
    NDArray[np.int16] | None,
    NDArray[np.int16] | None,
    NDArray[np.int16] | None,
]:
    img_path = paths["data"] / "Images" / lq_id[0:6]
    seg_path = paths["data"] / "Segmentations" / lq_id[0:6]
    nce_id = [p.stem for p in img_path.glob(f"{lq_id[0:6]}HQ*")][0]

    ac_or_hq_candidates = [p.stem for p in seg_path.glob(f"{lq_id[0:6]}{ac_or_hq}*")]
    ac_or_hq_candidates = sorted(
        ac_or_hq_candidates,
        key=lambda x: abs(int(x[8:11]) - int(lq_id[8:11])),
    )
    ac_or_hq_id = ac_or_hq_candidates[0]

    nce_img = itk.ReadImage(str(img_path / f"{nce_id}.nrrd"))
    ac_or_hq_img = itk.ReadImage(str(seg_path / f"{ac_or_hq_id}.nrrd"))
    lq_img = itk.ReadImage(str(seg_path / f"{lq_id}-label.nrrd"))
    pred_img = itk.ReadImage(str(paths["predictions"] / f"{lq_id}-label.nrrd"))

    ac_or_hq_img = itk.Resample(ac_or_hq_img, nce_img, defaultPixelValue=-1)
    lq_img = itk.Resample(lq_img, nce_img, defaultPixelValue=-1)
    pred_img = itk.Resample(pred_img, nce_img, defaultPixelValue=-1)

    transform_path = paths["transforms"] / img_path.stem
    hq_transform_candidates = list(transform_path.glob(f"{ac_or_hq_id[8:11]}_to_*.h5"))
    if len(hq_transform_candidates) > 1:
        raise ValueError(hq_transform_candidates)

    if len(hq_transform_candidates) == 1:
        hq_transform = itk.ReadTransform(str(hq_transform_candidates[0]))
        ac_or_hq_img = itk.Resample(ac_or_hq_img, hq_transform, defaultPixelValue=-1)
        lq_img = itk.Resample(lq_img, hq_transform, defaultPixelValue=-1)
        # pred_img = itk.Resample(pred_img, hq_transform, defaultPixelValue=-1)

    try:
        lq_img, lq_lower, lq_upper = trim_lq(lq_img)
        pred_img, pred_lower, pred_upper = trim_lq(pred_img)
    except (IndexError, ValueError):
        return None, None, None

    lq_img = itk.GetArrayFromImage(lq_img)[lq_lower:lq_upper, :, :]
    pred_img = itk.GetArrayFromImage(pred_img)[pred_lower:pred_upper, :, :]
    ac_or_hq_img = itk.GetArrayFromImage(ac_or_hq_img)[pred_lower:pred_upper, :, :]
    lq_img[lq_img == -1] = 0
    pred_img[pred_img == -1] = 0
    ac_or_hq_img[ac_or_hq_img == -1] = 0

    return lq_img, ac_or_hq_img, pred_img


# -------------------------------------------------------------------------


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", "-d", help="Data path", type=str)
    parser.add_argument("--subset", "-su", help="Data subset", type=str)
    arguments = parser.parse_args()

    paths = {}
    paths["data"] = Path(arguments.data)
    paths["predictions"] = Path(arguments.path)
    paths["transforms"] = Path(arguments.data) / "Transforms"
    task = paths["predictions"].stem.split("-")[1]
    model_name = paths["predictions"].parent.stem

    # Create dataframe if not present
    df_path = paths["predictions"].parents[1] / f"{task}_test_segmentations.csv"

    try:
        df = pd.read_csv(df_path, index_col=0, header=0)

    except FileNotFoundError:
        df = pd.DataFrame(columns=["LQ"])

    for nce_path in paths["predictions"].glob("*"):
        nce_id = nce_path.stem.split("-")[0]
        if nce_id in ["T107A0LQ067", "T115A0LQ151"]:
            continue

        print(nce_id)

        if task == Task.JOINT:
            in_img, gt_img, pred_img = load_and_transform(nce_id, paths, "AC")
        elif task == Task.SUPER_RES:
            in_img, gt_img, pred_img = load_and_transform(nce_id, paths, "HQ")

        if in_img is None:
            continue

        if nce_id not in df.index:
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index=[nce_id], columns=["LQ"])],
                axis=0,
            )

        if np.isnan(df.loc[nce_id, "LQ"]):
            df.loc[nce_id, "LQ"] = calc_dice(in_img, gt_img)

        df.loc[nce_id, model_name] = calc_dice(pred_img, gt_img)

    df.to_csv(df_path, index=True)


if __name__ == "__main__":
    main()
