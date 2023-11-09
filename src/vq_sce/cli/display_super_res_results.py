import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as itk  # noqa: N813

from vq_sce import ABDO_WINDOW, HU_MAX, HU_MIN, MIN_HQ_DEPTH
from vq_sce.preproc.preprocess import HU_DEFAULT, HU_THRESHOLD

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


def load_and_transform(
    lq_id: str,
    paths: dict[str, Path],
    ignore: list[str],
) -> tuple[itk.Image, itk.Image, itk.Image]:
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

    ignore_dir = Path(__file__).resolve().parents[1] / "preproc"

    with open(ignore_dir / "ignore.json") as fp:
        ignore = json.load(fp)["image_ignore"]

    paths = {}
    paths["data"] = Path(arguments.data) / "Images"
    base_results_path = Path(arguments.path)

    first_expt_path_candidates = list(
        (base_results_path / expt_list[0]).glob("predictions*"),
    )
    assert len(first_expt_path_candidates) == 1
    nce_paths = list(first_expt_path_candidates[0].glob("*"))

    if arguments.save:
        save_path = first_expt_path_candidates[0].parents[1] / "output_super_res"
        save_path.mkdir(exist_ok=True)

    for nce_path in nce_paths:
        nce_id = nce_path.stem
        expt_imgs = {}

        for expt in expt_list:
            expt_path_candidates = list(
                (base_results_path / expt).glob("predictions-super_res"),
            )
            assert len(expt_path_candidates) == 1, expt_path_candidates
            paths["predictions"] = expt_path_candidates[0]

            lq_img, hq_img, pred_img = load_and_transform(nce_id, paths, ignore)
            lq_np = itk.GetArrayFromImage(lq_img)
            expt_imgs[expt] = itk.GetArrayFromImage(pred_img)
            hq_np = itk.GetArrayFromImage(hq_img)

        z = MIN_HQ_DEPTH // 2 - 1

        plt.figure(figsize=(18, 12))

        plt.subplot(2, num_expts, 1)
        plt.imshow(lq_np[z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW)
        plt.axis("off")
        plt.title(f"LQ: {z}")
        plt.subplot(2, num_expts, num_expts)
        plt.imshow(hq_np[z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW)
        plt.axis("off")
        plt.title(f"HQ: {nce_id}")

        for (
            i,
            expt,
        ) in enumerate(expt_list):
            plt.subplot(2, num_expts, num_expts + i + 1)
            plt.imshow(expt_imgs[expt][z, 128:384, 128:384], cmap="bone", **ABDO_WINDOW)
            plt.axis("off")
            plt.title(expt)

        if arguments.save:
            plt.savefig(save_path / f"{nce_id}_{z}_.png")
            plt.close()

        else:
            plt.show()
