import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as itk  # noqa: N813
from numpy.typing import NDArray
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from vq_sce import HU_MAX, HU_MIN, LQ_DEPTH, LQ_SLICE_THICK
from vq_sce.utils.dataloaders.build_dataloader import Subsets

METRICS = ["L1", "MSE", "pSNR", "SSIM"]

# -------------------------------------------------------------------------


def load_and_transform(
    lq_id: str,
    paths: dict[str, Path],
    source_coords: dict[str, dict[str, list[int]]],
) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
    data_path = paths["data"]
    hq_candidates = [p.stem for p in data_path.glob(f"{lq_id[0:6]}*.npy")]

    hq_candidates = sorted(
        hq_candidates,
        key=lambda x: abs(int(x[-3:]) - int(lq_id[-3:])),
    )
    hq_id = hq_candidates[0]

    lq_coords = list(source_coords[lq_id].values())[0]
    hq_coords = list(source_coords[hq_id].values())[0]
    hq_coords = [
        lq_coords[0] - hq_coords[0],
        lq_coords[0] - hq_coords[0] + (LQ_SLICE_THICK * LQ_DEPTH),
    ]

    hq_img = np.load(str(data_path / f"{hq_id}.npy"))
    pred_img = itk.ReadImage(str(paths["predictions"] / f"{lq_id}.nrrd"))

    hu_filter = itk.ClampImageFilter()
    hu_filter.SetLowerBound(HU_MIN)
    hu_filter.SetUpperBound(HU_MAX)
    pred_img = hu_filter.Execute(pred_img)

    pred_img_np = itk.GetArrayFromImage(pred_img).astype(np.float16)
    hq_img = hq_img[hq_coords[0] : hq_coords[1], :, :]

    return hq_img, pred_img_np


# -------------------------------------------------------------------------


def calc_global_metrics(
    ce_volume: NDArray[np.float16],
    pred_volume: NDArray[np.float16],
) -> dict[str, float]:
    """Calculate L1, MSE, pSNR, SSIM between predicted and ground truth image."""
    metrics = {
        "MSE": float(mean_squared_error(ce_volume, pred_volume)),
        "pSNR": float(
            peak_signal_noise_ratio(
                ce_volume,
                pred_volume,
                data_range=HU_MAX - HU_MIN,
            ),
        ),
        "SSIM": float(
            structural_similarity(
                ce_volume,
                pred_volume,
                data_range=HU_MAX - HU_MIN,
            ),
        ),
    }
    metrics["L1"] = float(np.mean(np.abs(ce_volume - pred_volume)))

    return metrics


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
    paths["bounding_boxes"] = Path(arguments.data) / "BoundingBoxes"

    if arguments.subset == Subsets.TEST:
        paths["data"] = paths["data"] / "test"
    else:
        paths["data"] = paths["data"] / "train"

    with open(paths["data"] / "source_coords.json") as fp:
        source_coords = json.load(fp)
    paths["data"] /= "HQ"

    model_name = paths["predictions"].parent.stem
    epochs = paths["predictions"].stem.split("-")[1]

    metrics: dict[str, list[str | float]] = {
        "id": [],
        "L1": [],
        "MSE": [],
        "pSNR": [],
        "SSIM": [],
        "focal_L1": [],
    }

    for nce_path in paths["predictions"].glob("*"):
        nce_id = nce_path.stem
        ce_img, pred_img = load_and_transform(nce_id, paths, source_coords)

        # Calculate global metrics
        subject_global_metrics = calc_global_metrics(ce_img, pred_img)
        metrics["id"].append(nce_id)

        for metric in METRICS:
            metrics[metric].append(subject_global_metrics[metric])

        metrics["focal_L1"].append(
            float(
                np.mean(
                    np.abs(ce_img[:, 192:320, 192:320] - pred_img[:, 192:320, 192:320]),
                ),
            ),
        )

    # Save global metrics
    csv_name = f"super_res_{arguments.subset}"

    # Create dataframe if not present
    df_path = paths["predictions"].parents[1] / f"{csv_name}.csv"

    try:
        df = pd.read_csv(df_path, index_col=0, header=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(
            index=metrics["id"],
            columns=pd.MultiIndex.from_product(
                [METRICS + ["focal_L1"], [f"{model_name}-{epochs}"]],
            ),
        )

    finally:
        for metric in METRICS:
            df[(metric, f"{model_name}-{epochs}")] = metrics[metric]

        df[("focal_L1", f"{model_name}-{epochs}")] = metrics["focal_L1"]
        df.to_csv(df_path, index=True)
