import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from vq_sce import ABDO_WINDOW, IMAGE_HEIGHT, LQ_SLICE_THICK, MIN_HQ_DEPTH


#-------------------------------------------------------------------------

def display_images(
    nce: np.ndarray,
    ce: np.ndarray,
    ce_coords: list[int],
    hq: np.ndarray,
    hq_coords: list[int],
    lq: np.ndarray,
    lq_coords: list[int],
    lq_name: str
) -> None:

    nce_level = lq_coords[0] + LQ_SLICE_THICK
    hq_level = nce_level - hq_coords[0]
    ce_level = nce_level - ce_coords[0]

    lq = np.repeat(lq, 4, axis=0)
    hq_lq_coords = [
        lq_coords[0] - hq_coords[0],
        lq_coords[0] - hq_coords[0] + MIN_HQ_DEPTH
    ]
    if hq_lq_coords[0] < 0 or hq_lq_coords[1] > hq.shape[0]:
        print("FAIL")
        return None

    plt.figure(figsize=(18, 8))
    plt.subplot(2, 6, 1)
    plt.imshow(nce[nce_level, ...], cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 2)
    plt.imshow(ce[ce_level, ...], cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 3)
    plt.imshow(
        ce[ce_level, ...] - nce[nce_level, ...],
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")

    plt.subplot(2, 6, 4)
    plt.imshow(hq[hq_level, ...], cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 5)
    plt.imshow(lq[1, ...], cmap="bone", **ABDO_WINDOW)
    plt.title(lq_name)
    plt.axis("off")
    plt.subplot(2, 6, 6)
    plt.imshow(
        hq[hq_level, ...] - lq[1, ...],
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")

    plt.subplot(2, 6, 7)
    plt.imshow(
        np.flipud(nce[ce_coords[0]:ce_coords[1], IMAGE_HEIGHT // 2, :]),
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 8)
    plt.imshow(np.flipud(ce[:, IMAGE_HEIGHT // 2, :]), cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 9)
    plt.imshow(
        np.flipud(ce[:, IMAGE_HEIGHT // 2, :]) - \
        np.flipud(nce[ce_coords[0]:ce_coords[1], IMAGE_HEIGHT // 2, :]),
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")

    plt.subplot(2, 6, 10)
    plt.imshow(
        np.flipud(hq[hq_lq_coords[0]:hq_lq_coords[1], IMAGE_HEIGHT // 2, :]),
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 11)
    plt.imshow(np.flipud(lq[:, IMAGE_HEIGHT // 2, :]), cmap="bone", **ABDO_WINDOW)
    plt.axis("off")
    plt.subplot(2, 6, 12)
    plt.imshow(
        np.flipud(lq[:, IMAGE_HEIGHT // 2, :]) - \
        np.flipud(hq[hq_lq_coords[0]:hq_lq_coords[1], IMAGE_HEIGHT // 2, :]),
        cmap="bone", **ABDO_WINDOW)
    plt.axis("off")

    plt.show()


#-------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_include", '-t', type=str, help="Include IDs")
    parser.add_argument("--start_at", '-sa', type=str, help="Start ID")
    parser.add_argument("--stop_before", '-sb', type=str, help="End ID")
    arguments = parser.parse_args()

    cwd = Path()
    ce_path = cwd / "CE"
    hq_path = cwd / "HQ"
    lq_path = cwd / "LQ"

    with open(cwd / "source_coords.json", 'r') as fp:
        source_coords = json.load(fp)

    if arguments.to_include is None:
        subject_ids = [sub.stem[0:6] for sub in ce_path.glob('*')]
    else:
        subject_ids = [sub.stem[0:6] for sub in ce_path.glob('*') \
                       if sub.stem[0:6] in arguments.to_include.split(',')]

    idx_start = None if arguments.start_at is None \
                     else subject_ids.index(arguments.start_at)
    idx_end = None if arguments.stop_before is None \
                   else subject_ids.index(arguments.stop_before)
    subject_ids = subject_ids[idx_start:idx_end]

    for subject_id in subject_ids:
        ce_name = list(ce_path.glob(f"{subject_id}*"))[0].stem
        ce_img = np.load(ce_path / f"{ce_name}.npy")

        nce_name = list(hq_path.glob(f"{subject_id}*.npy"))[0]
        nce_img = np.load(nce_name)
        ce_coords = source_coords[ce_name][nce_name.stem]

        lq_names = list(lq_path.glob(f"{subject_id}*.npy"))
        hq_names = list(hq_path.glob(f"{subject_id}*.npy"))
        hq_names = [n.stem for n in hq_names]

        for lq_name in lq_names:
            lq_img = np.load(lq_name)
            lq_coords = source_coords[lq_name.stem][nce_name.stem]

            hq_candidates = sorted(
                hq_names, key=lambda x: abs(int(x[-3:]) - int(lq_name.stem[-3:]))
            )
            hq_name = hq_candidates[0]
            hq_img = np.load(hq_path / f"{hq_name}.npy")
            hq_coords = source_coords[hq_name][nce_name.stem]

            display_images(
                nce=nce_img,
                ce=ce_img,
                ce_coords=ce_coords,
                hq=hq_img,
                hq_coords=hq_coords,
                lq=lq_img,
                lq_coords=lq_coords,
                lq_name=lq_name.stem
            )


#-------------------------------------------------------------------------
""" Check images saved by preprocessing pipeline """

if __name__ == "__main__":
    main()
