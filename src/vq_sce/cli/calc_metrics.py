import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as itk  # noqa: N813
from numpy.typing import NDArray
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from vq_sce import HU_MAX, HU_MIN
from vq_sce.preproc.preprocess import HU_DEFAULT

METRICS = ["L1", "MSE", "pSNR", "SSIM"]

# -------------------------------------------------------------------------


def load_and_transform(
    nce_id: str,
    paths: dict[str, Path],
) -> tuple[itk.Image, itk.Image]:
    data_path = paths["data"] / nce_id[0:6]
    ce_candidates = [p.stem for p in data_path.glob(f"{nce_id[0:6]}AC*")]

    ce_candidates = sorted(
        ce_candidates,
        key=lambda x: abs(int(x[-3:]) - int(nce_id[-3:])),
    )
    ce_id = ce_candidates[0]

    ce_img = itk.ReadImage(str(data_path / f"{ce_id}.nrrd"))
    pred_img = itk.ReadImage(str(paths["predictions"] / f"{nce_id}.nrrd"))
    ce_img = itk.Resample(ce_img, pred_img, defaultPixelValue=HU_DEFAULT)

    hu_filter = itk.ClampImageFilter()
    hu_filter.SetLowerBound(HU_MIN)
    hu_filter.SetUpperBound(HU_MAX)
    pred_img = hu_filter.Execute(pred_img)
    ce_img = hu_filter.Execute(ce_img)

    transform_path = paths["transforms"] / data_path.stem

    ce_transform_candidates = list(transform_path.glob(f"{ce_id[-3:]}_to_*.h5"))
    if len(ce_transform_candidates) > 1:
        raise ValueError(ce_transform_candidates)

    if len(ce_transform_candidates) == 1:
        ce_transform = itk.ReadTransform(str(ce_transform_candidates[0]))
        ce_img = itk.Resample(ce_img, ce_transform, defaultPixelValue=HU_DEFAULT)

    return ce_img, pred_img


# -------------------------------------------------------------------------


def get_bounding_boxes(
    ce_img: itk.Image,
    pred_img: itk.Image,
    bounding_box_data: list[str],
) -> tuple[dict[str, NDArray[np.int16]], dict[str, NDArray[np.int16]]]:
    fiduciaries: dict[str, Any] = {"RK": {}, "LK": {}, "VC": {}, "AO": {}}
    for region in fiduciaries:
        fiduciaries[region]["L"] = np.zeros(2)
        fiduciaries[region]["P"] = np.zeros(2)

    for line in bounding_box_data[3:]:
        fiduciary = line.split(",")
        region = fiduciary[11][0:2]
        side = fiduciary[11][-1]

        if side == "R":
            fiduciaries[region]["L"][0] = -float(fiduciary[1])
        elif side == "L":
            fiduciaries[region]["L"][1] = -float(fiduciary[1])
        elif side == "A":
            fiduciaries[region]["P"][0] = -float(fiduciary[2])
        else:
            fiduciaries[region]["P"][1] = -float(fiduciary[2])

        fiduciaries[region]["S"] = float(fiduciary[3])

    bounds: dict[str, Any] = {"RK": {}, "LK": {}, "VC": {}, "AO": {}}

    for region in fiduciaries.keys():
        xs = (
            fiduciaries[region]["L"] - pred_img.GetOrigin()[0]
        ) / pred_img.GetSpacing()[0]
        ys = (
            fiduciaries[region]["P"] - pred_img.GetOrigin()[1]
        ) / pred_img.GetSpacing()[1]

        try:
            zs = (
                fiduciaries[region]["S"] - pred_img.GetOrigin()[2]
            ) / pred_img.GetSpacing()[2]
        except KeyError:
            del bounds[region]
            continue

        bounds[region]["x"] = slice(*np.round(xs).astype(np.int16))
        bounds[region]["y"] = slice(*np.round(ys).astype(np.int16))
        bounds[region]["z"] = np.round(zs).astype(np.int16)

    pred_np = itk.GetArrayFromImage(pred_img)
    ce_np = itk.GetArrayFromImage(ce_img)

    ce_sub_volumes = dict.fromkeys(bounds.keys())
    pred_sub_volumes = dict.fromkeys(bounds.keys())

    for region in ce_sub_volumes.keys():
        roi_bounds = bounds[region]
        ce_sub_volumes[region] = ce_np[
            roi_bounds["z"] - 8 : roi_bounds["z"] + 8,
            roi_bounds["y"],
            roi_bounds["x"],
        ]
        pred_sub_volumes[region] = pred_np[
            roi_bounds["z"] - 8 : roi_bounds["z"] + 8,
            roi_bounds["y"],
            roi_bounds["x"],
        ]

    return ce_sub_volumes, pred_sub_volumes


# -------------------------------------------------------------------------


def calc_global_metrics(
    ce_volume: itk.Image,
    pred_volume: itk.Image,
) -> dict[str, float]:
    """Calculate L1, MSE, pSNR, SSIM between predicted and ground truth image."""
    ce_volume_np = itk.GetArrayFromImage(ce_volume)
    pred_volume_np = itk.GetArrayFromImage(pred_volume)

    metrics = {
        "MSE": float(mean_squared_error(ce_volume_np, pred_volume_np)),
        "pSNR": float(
            peak_signal_noise_ratio(
                ce_volume_np,
                pred_volume_np,
                data_range=HU_MAX - HU_MIN,
            ),
        ),
        "SSIM": float(
            structural_similarity(
                ce_volume_np,
                pred_volume_np,
                data_range=HU_MAX - HU_MIN,
            ),
        ),
    }
    metrics["L1"] = float(np.mean(np.abs(ce_volume_np - pred_volume_np)))

    return metrics


# -------------------------------------------------------------------------


def calc_intensity_diffs(
    ce_sub_volume: dict[str, NDArray[np.int16]],
    pred_sub_volume: dict[str, NDArray[np.int16]],
) -> dict[str, float]:
    results = {}

    for region in ce_sub_volume.keys():
        results[region] = np.mean(
            np.abs(ce_sub_volume[region] - pred_sub_volume[region]),
        )

    return results


# -------------------------------------------------------------------------


def calc_intensities(pred_sub_volume: dict[str, NDArray[np.int16]]) -> dict[str, float]:
    results = {}

    for region in pred_sub_volume.keys():
        results[region] = float(np.mean(pred_sub_volume[region]))

    return results


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

    model_name = paths["predictions"].parent.stem
    epochs = paths["predictions"].stem.split("-")[1]

    global_metrics: dict[str, list[str | float]] = {
        "id": [],
        "L1": [],
        "MSE": [],
        "pSNR": [],
        "SSIM": [],
    }

    intensity_diffs: dict[str, list[str | float]] = {
        "id": [],
        "RK": [],
        "LK": [],
        "VC": [],
        "AO": [],
    }
    intensities: dict[str, list[str | float]] = {
        "id": [],
        "RK": [],
        "LK": [],
        "VC": [],
        "AO": [],
    }
    regions = ["RK", "LK", "VC", "AO"]

    for nce_path in paths["predictions"].glob("*"):
        nce_id = nce_path.stem

        with open(paths["bounding_boxes"] / f"{nce_id[0:6]}.fcsv") as fp:
            bounding_box = fp.readlines()

        ce_img, pred_img = load_and_transform(nce_id, paths)
        ce_sub, pred_sub = get_bounding_boxes(ce_img, pred_img, bounding_box)

        # Calculate global metrics
        subject_global_metrics = calc_global_metrics(ce_img, pred_img)
        global_metrics["id"].append(nce_id)

        for metric in METRICS:
            global_metrics[metric].append(subject_global_metrics[metric])

        # Calculate bounding box L1
        subject_intensity_diffs = calc_intensity_diffs(ce_sub, pred_sub)
        intensity_diffs["id"].append(nce_id)

        for region in regions:
            try:
                intensity_diffs[region].append(subject_intensity_diffs[region])

            except KeyError:
                intensity_diffs[region].append(None)

        # Calculate bounding box intensities
        subject_intensities = calc_intensities(pred_sub)
        intensities["id"].append(nce_id)

        for region in regions:
            try:
                intensities[region].append(subject_intensities[region])

            except KeyError:
                intensities[region].append(None)

    # Save global metrics
    csv_name = f"contrast_{arguments.subset}_global"

    # Create dataframe if not present
    df_path = paths["predictions"].parents[1] / f"{csv_name}.csv"

    try:
        df = pd.read_csv(df_path, index_col=0, header=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(
            index=global_metrics["id"],
            columns=pd.MultiIndex.from_product([METRICS, [f"{model_name}-{epochs}"]]),
        )

    finally:
        for metric in METRICS:
            df[(metric, f"{model_name}-{epochs}")] = global_metrics[metric]

        df.to_csv(df_path, index=True)

    # Save bounding box L1
    csv_name = f"contrast_{arguments.subset}_focal_L1"
    df_path = paths["predictions"].parents[1] / f"{csv_name}.csv"

    try:
        df = pd.read_csv(df_path, index_col=0, header=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(
            index=intensity_diffs["id"],
            columns=pd.MultiIndex.from_product([regions, [f"{model_name}-{epochs}"]]),
        )

    finally:
        for region in regions:
            df[(region, f"{model_name}-{epochs}")] = intensity_diffs[region]

        df.to_csv(df_path, index=True)

    # Save bounding box intensities
    csv_name = f"contrast_{arguments.subset}_focal_HU"
    df_path = paths["predictions"].parents[1] / f"{csv_name}.csv"

    try:
        df = pd.read_csv(df_path, index_col=0, header=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(
            index=intensities["id"],
            columns=pd.MultiIndex.from_product([regions, [f"{model_name}-{epochs}"]]),
        )
    finally:
        for region in regions:
            df[(region, f"{model_name}-{epochs}")] = intensities[region]

        df.to_csv(df_path, index=True)
