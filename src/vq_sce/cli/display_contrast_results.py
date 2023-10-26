import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as itk  # noqa: N813

from vq_sce import ABDO_WINDOW, HU_MAX, HU_MIN
from vq_sce.preproc.preprocess import HU_DEFAULT

# -------------------------------------------------------------------------


def load_and_transform(
    nce_id: str,
    paths: dict[str, Path],
) -> tuple[itk.Image, itk.Image, itk.Image]:
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


def get_roi_z(bounding_box_data: list[str]) -> dict[str, float]:
    fiduciaries = {}

    for line in bounding_box_data[3:]:
        fiduciary = line.split(",")
        region = fiduciary[11][0:2]
        fiduciaries[region] = float(fiduciary[3])

    return fiduciaries


# -------------------------------------------------------------------------


def main() -> None:
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--expts", "-e", help="Expt list", type=str)
    parser.add_argument("--data", "-d", help="Data path", type=str)
    parser.add_argument("--save", "-s", help="Save images", action="store_true")
    arguments = parser.parse_args()

    expt_list = arguments.expts.split(",")
    num_expts = len(expt_list)

    paths = {}
    paths["data"] = Path(arguments.data) / "Images"
    base_results_path = Path(arguments.path)
    paths["transforms"] = Path(arguments.data) / "Transforms"
    paths["bounding_boxes"] = Path(arguments.data) / "BoundingBoxes"

    first_expt_path_candidates = list(
        (base_results_path / expt_list[0]).glob("predictions*"),
    )
    assert len(first_expt_path_candidates) == 1
    nce_paths = list(first_expt_path_candidates[0].glob("*"))

    if arguments.save:
        save_path = first_expt_path_candidates[0].parents[1] / "output_contrast"
        save_path.mkdir(exist_ok=True)

    for nce_path in nce_paths:
        nce_id = nce_path.stem
        expt_imgs = {}

        with open(paths["bounding_boxes"] / f"{nce_id[0:6]}.fcsv") as fp:
            bounding_box = fp.readlines()

        fiduciaries = get_roi_z(bounding_box)

        for expt in expt_list:
            expt_path_candidates = list(
                (base_results_path / expt).glob("predictions-contrast"),
            )
            assert len(expt_path_candidates) == 1, expt_path_candidates
            paths["predictions"] = expt_path_candidates[0]

            nce_img, ce_img, pred_img = load_and_transform(nce_id, paths)

            nce_np = itk.GetArrayFromImage(nce_img)
            expt_imgs[expt] = itk.GetArrayFromImage(pred_img)
            ce_np = itk.GetArrayFromImage(ce_img)

        for z_vals in fiduciaries.values():
            z = np.round(
                z_vals - pred_img.GetOrigin()[2] / pred_img.GetSpacing()[2],
            ).astype(np.int16)

            plt.figure(figsize=(18, 12))

            plt.subplot(2, num_expts, 1)
            plt.imshow(nce_np[z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW)
            plt.axis("off")
            plt.title(f"NCE: {z}")
            plt.subplot(2, num_expts, num_expts)
            plt.imshow(ce_np[z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW)
            plt.axis("off")
            plt.title(f"CE: {nce_id}")

            for (
                i,
                expt,
            ) in enumerate(expt_list):
                plt.subplot(2, num_expts, num_expts + i + 1)
                plt.imshow(
                    expt_imgs[expt][z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW
                )
                plt.axis("off")
                plt.title(expt)

            if arguments.save:
                plt.savefig(save_path / f"{nce_id}_{z}_.png")
                plt.close()

            else:
                plt.show()
