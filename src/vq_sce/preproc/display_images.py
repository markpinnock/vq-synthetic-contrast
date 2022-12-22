import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ABDO_WINDOW = {"vmin": -150, "vmax": 250}
IMAGE_HEIGHT = 512
LQ_DEPTH = 3


#-------------------------------------------------------------------------

def display_images(
    img1: np.ndarray,
    img2: np.ndarray,
    name1: str,
    name2: str
) -> None:

    if img1.shape[0] == LQ_DEPTH:
        img1 = np.repeat(img1, 4, axis=0)

    mid_point = img1.shape[0] // 2

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(img1[mid_point, ...], cmap="bone", **ABDO_WINDOW)
    plt.title(name1)
    plt.subplot(2, 3, 2)
    plt.imshow(img2[mid_point, ...], cmap="bone", **ABDO_WINDOW)
    plt.title(name2)
    plt.subplot(2, 3, 3)
    plt.imshow(
        img1[mid_point, ...] - img2[mid_point, ...],
        cmap="bone", **ABDO_WINDOW
    )
    plt.subplot(2, 3, 4)
    plt.imshow(img1[:, IMAGE_HEIGHT // 2, :], cmap="bone", **ABDO_WINDOW)
    plt.subplot(2, 3, 5)
    plt.imshow(img2[:, IMAGE_HEIGHT // 2, :], cmap="bone", **ABDO_WINDOW)
    plt.subplot(2, 3, 6)
    plt.imshow(
        img1[:, IMAGE_HEIGHT // 2, :] - img2[:, IMAGE_HEIGHT // 2, :],
        cmap="bone", **ABDO_WINDOW
    )
    plt.show()


#-------------------------------------------------------------------------

def main() -> None:
    cwd = Path()
    ce_path = cwd / "CE"
    hq_path = cwd / "HQ"
    lq_path = cwd / "LQ"

    with open(cwd / "source_coords.json", 'r') as fp:
        source_coords = json.load(fp)

    for ce_name in ce_path.glob('*'):
        subject_id = ce_name.stem[0:6]
        ce_img = np.load(ce_name)

        hq_name = list(hq_path.glob(f"{subject_id}*.npy"))[0]
        hq_img = np.load(hq_name)
        ce_coords = source_coords[ce_name.stem][hq_name.stem]
        hq_img = hq_img[ce_coords[0]:ce_coords[1], ...]

        assert ce_img.shape == hq_img.shape
        display_images(ce_img, hq_img, ce_name.stem, hq_name.stem)

        lq_names = list(lq_path.glob(f"{subject_id}*.npy"))

        for lq_name in lq_names:
            lq_img = np.load(lq_name)
            hq_candidates = list(source_coords[lq_name.stem].keys())

            hq_name = hq_candidates[0]
            hq_img = np.load(hq_path / f"{hq_name}.npy")
            lq_coords = source_coords[lq_name.stem][hq_name]
            hq_img = hq_img[lq_coords[0]:lq_coords[1], ...]

            display_images(lq_img, hq_img, lq_name.stem, hq_name)


#-------------------------------------------------------------------------
""" Check images saved by preprocessing pipeline """

if __name__ == "__main__":
    main()
