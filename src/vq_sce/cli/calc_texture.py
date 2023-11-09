import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as itk  # noqa: N813
from numpy.typing import NDArray
from radiomics import featureextractor

from vq_sce import HU_MAX, HU_MIN, MIN_HQ_DEPTH
from vq_sce.networks.model import Task
from vq_sce.preproc.preprocess import HU_DEFAULT, HU_THRESHOLD

# -------------------------------------------------------------------------


def load_and_transform_ce(
    nce_id: str,
    paths: dict[str, Path],
) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]:
    data_path = paths["data"] / nce_id[0:6]
    ce_candidates = [p.stem for p in data_path.glob(f"{nce_id[0:6]}AC*")]

    ce_candidates = sorted(
        ce_candidates,
        key=lambda x: abs(int(x[-3:]) - int(nce_id[-3:])),
    )
    ce_id = ce_candidates[0]

    ce_img = itk.ReadImage(str(data_path / f"{ce_id}.nrrd"))
    nce_img = itk.ReadImage(str(data_path / f"{nce_id}.nrrd"))
    pred_img = itk.ReadImage(str(paths["predictions"] / f"{nce_id}.nrrd"))

    ce_img = itk.Resample(ce_img, nce_img, defaultPixelValue=HU_DEFAULT)
    pred_img = itk.Resample(pred_img, nce_img, defaultPixelValue=HU_DEFAULT)

    hu_filter = itk.ClampImageFilter()
    hu_filter.SetLowerBound(HU_MIN)
    hu_filter.SetUpperBound(HU_MAX)
    nce_img = hu_filter.Execute(nce_img)
    pred_img = hu_filter.Execute(pred_img)
    ce_img = hu_filter.Execute(ce_img)

    transform_path = paths["transforms"] / data_path.stem

    nce_transform_candidates = list(transform_path.glob(f"{nce_id[-3:]}_to_*.h5"))
    if len(nce_transform_candidates) > 1:
        raise ValueError(nce_transform_candidates)

    if len(nce_transform_candidates) == 1:
        nce_transform = itk.ReadTransform(str(nce_transform_candidates[0]))
        nce_img = itk.Resample(nce_img, nce_transform, defaultPixelValue=HU_DEFAULT)

    ce_transform_candidates = list(transform_path.glob(f"{ce_id[-3:]}_to_*.h5"))
    if len(ce_transform_candidates) > 1:
        raise ValueError(ce_transform_candidates)

    if len(ce_transform_candidates) == 1:
        ce_transform = itk.ReadTransform(str(ce_transform_candidates[0]))
        ce_img = itk.Resample(ce_img, ce_transform, defaultPixelValue=HU_DEFAULT)

    return nce_img, ce_img, pred_img


# -------------------------------------------------------------------------


def trim_lq(source: itk.Image) -> tuple[itk.Image, int, int]:
    source_slice_means = itk.GetArrayFromImage(source).mean(axis=(1, 2))
    source_slice_idx = np.argwhere(source_slice_means > HU_THRESHOLD)
    source_lower = int(source_slice_idx[0])
    source_upper = int(source_slice_idx[-1]) + 1

    if (source_upper - source_lower) != MIN_HQ_DEPTH:
        raise ValueError(source_lower, source_upper)

    return source, source_lower, source_upper


# -------------------------------------------------------------------------


def load_and_transform_sr(
    lq_id: str,
    paths: dict[str, Path],
    ignore: list[str],
) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]:
    data_path = paths["data"] / lq_id[0:6]
    nce_id = [
        p.stem for p in data_path.glob(f"{lq_id[0:6]}HQ*") if p.stem not in ignore
    ][0]

    hq_candidates = [p.stem for p in data_path.glob(f"{lq_id[0:6]}HQ*")]
    hq_candidates = sorted(
        hq_candidates,
        key=lambda x: abs(int(x[-3:]) - int(lq_id[-3:])),
    )
    hq_id = hq_candidates[0]

    nce_img = itk.ReadImage(str(data_path / f"{nce_id}.nrrd"))
    hq_img = itk.ReadImage(str(data_path / f"{hq_id}.nrrd"))
    lq_img = itk.ReadImage(str(data_path / f"{lq_id}.nrrd"))
    pred_img = itk.ReadImage(str(paths["predictions"] / f"{lq_id}.nrrd"))

    hq_img = itk.Resample(hq_img, nce_img, defaultPixelValue=HU_DEFAULT)
    lq_img = itk.Resample(lq_img, nce_img, defaultPixelValue=HU_DEFAULT)
    pred_img = itk.Resample(pred_img, nce_img, defaultPixelValue=HU_DEFAULT)

    transform_path = paths["transforms"] / data_path.stem
    hq_transform_candidates = list(transform_path.glob(f"{hq_id[-3:]}_to_*.h5"))
    if len(hq_transform_candidates) > 1:
        raise ValueError(hq_transform_candidates)

    if len(hq_transform_candidates) == 1:
        hq_transform = itk.ReadTransform(str(hq_transform_candidates[0]))
        hq_img = itk.Resample(hq_img, hq_transform, defaultPixelValue=HU_DEFAULT)
        lq_img = itk.Resample(lq_img, hq_transform, defaultPixelValue=HU_DEFAULT)

    lq_img, lq_lower, lq_upper = trim_lq(lq_img)
    pred_img, pred_lower, pred_upper = trim_lq(pred_img)

    hu_filter = itk.ClampImageFilter()
    hu_filter.SetLowerBound(HU_MIN)
    hu_filter.SetUpperBound(HU_MAX)

    lq_img = hu_filter.Execute(lq_img)[:, :, lq_lower:lq_upper]
    pred_img = hu_filter.Execute(pred_img)[:, :, pred_lower:pred_upper]
    hq_img = hu_filter.Execute(hq_img)[:, :, pred_lower:pred_upper]

    return lq_img, hq_img, pred_img


# -------------------------------------------------------------------------


def get_bounding_boxes(
    in_img: itk.Image,
    gt_img: itk.Image,
    pred_img: itk.Image,
    bounding_box_data: list[str],
) -> tuple[itk.Image, itk.Image, itk.Image, itk.Image]:
    fiduciaries = {}
    fiduciaries["L"] = np.zeros(2)
    fiduciaries["P"] = np.zeros(2)

    for line in bounding_box_data[3:]:
        fiduciary = line.split(",")
        region = fiduciary[11][0:2]
        side = fiduciary[11][-1]

        if region == "LI":
            if side == "R":
                fiduciaries["L"][0] = -float(fiduciary[1])
            elif side == "L":
                fiduciaries["L"][1] = -float(fiduciary[1])
            elif side == "A":
                fiduciaries["P"][0] = -float(fiduciary[2])
            else:
                fiduciaries["P"][1] = -float(fiduciary[2])

            fiduciaries["S"] = float(fiduciary[3])

    bounds = {}

    xs = (fiduciaries["L"] - pred_img.GetOrigin()[0]) / pred_img.GetSpacing()[0]
    ys = (fiduciaries["P"] - pred_img.GetOrigin()[1]) / pred_img.GetSpacing()[1]
    zs = (fiduciaries["S"] - pred_img.GetOrigin()[2]) / pred_img.GetSpacing()[2]

    bounds["x"] = slice(*np.round(xs).astype(np.int16))
    bounds["y"] = slice(*np.round(ys).astype(np.int16))
    bounds["z"] = np.round(zs).astype(np.int16)

    mask_np = np.zeros([in_img.GetDepth()] + list(in_img.GetSize()[0:2])).astype(
        np.int16,
    )

    if pred_img.GetDepth() == MIN_HQ_DEPTH:
        mask_np[:, bounds["y"], bounds["x"]] = 1
    else:
        mask_np[bounds["z"] - 6 : bounds["z"] + 6, bounds["y"], bounds["x"]] = 1

    mask = itk.GetImageFromArray(mask_np)
    mask.SetDirection(in_img.GetDirection())
    mask.SetSpacing(in_img.GetSpacing())
    mask.SetOrigin(in_img.GetOrigin())

    return in_img, gt_img, pred_img, mask


# -------------------------------------------------------------------------


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--data", "-d", help="Data path", type=str)
    parser.add_argument("--subset", "-su", help="Data subset", type=str)
    arguments = parser.parse_args()

    paths = {}
    paths["data"] = Path(arguments.data) / "Images"
    paths["predictions"] = Path(arguments.path)
    paths["transforms"] = Path(arguments.data) / "Transforms"
    paths["bounding_boxes"] = Path(arguments.data) / "BoundingBoxes"
    task = paths["predictions"].stem.split("-")[1]

    with open(Path(__file__).parents[1] / "preproc" / "ignore.json") as fp:
        ignore = json.load(fp)["image_ignore"]

    with open(
        Path(__file__).parents[1] / "preproc" / "ignore_bounding_box.json",
    ) as fp:
        ignore += json.load(fp)

    model_name = paths["predictions"].parent.stem

    # Create dataframe if not present
    df_path = paths["predictions"].parents[1] / f"{task}_test_texture.csv"
    cols = ["NCE", "CE"] if task == Task.CONTRAST else ["LQ", "HQ"]

    try:
        df = pd.read_csv(df_path, index_col=0, header=0)

    except FileNotFoundError:
        df = pd.DataFrame(columns=cols)

    for nce_path in paths["predictions"].glob("*"):
        nce_id = nce_path.stem
        if nce_id in ignore:
            continue

        if nce_id not in df.index:
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index=[nce_id], columns=cols)],
                axis=0,
            )

        with open(paths["bounding_boxes"] / f"{nce_id[0:6]}.fcsv") as fp:
            bounding_box = fp.readlines()

        if task == Task.CONTRAST:
            in_img, gt_img, pred_img = load_and_transform_ce(nce_id, paths)
        elif task == Task.SUPER_RES:
            in_img, gt_img, pred_img = load_and_transform_sr(nce_id, paths, ignore)

        if in_img is None:
            continue

        in_img, gt_img, pred_img, mask = get_bounding_boxes(
            in_img,
            gt_img,
            pred_img,
            bounding_box,
        )

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("glcm")

        if np.isnan(df.loc[nce_id, cols[0]]):
            df.loc[nce_id, cols[0]] = extractor.execute(in_img, mask)[
                "original_glcm_Autocorrelation"
            ]

        if np.isnan(df.loc[nce_id, cols[1]]):
            df.loc[nce_id, cols[1]] = extractor.execute(gt_img, mask)[
                "original_glcm_Autocorrelation"
            ]

        df.loc[nce_id, model_name] = extractor.execute(pred_img, mask)[
            "original_glcm_Autocorrelation"
        ]

    df.to_csv(df_path, index=True)


if __name__ == "__main__":
    main()
