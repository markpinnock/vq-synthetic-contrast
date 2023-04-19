import argparse
import json
import os
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as itk  # noqa: N813

from vq_sce import (
    HQ_SLICE_THICK,
    HU_MAX,
    HU_MIN,
    LQ_DEPTH,
    LQ_SLICE_THICK,
    MIN_HQ_DEPTH,
)

HU_DEFAULT = -2048
HU_THRESHOLD = -2000

TypeImageDict = tuple[
    dict[str, itk.Image],
    dict[str, itk.Image],
    dict[str, itk.Image],
    dict[str, Path],
]


# -------------------------------------------------------------------------


class IgnoreType(TypedDict):
    subject_ignore: dict[str, str]
    image_ignore: list[str]


# -------------------------------------------------------------------------


class ImgConv:
    def __init__(
        self,
        file_path: str,
        save_path: str,
        include: list[str] | None = None,
        ignore: IgnoreType | None = None,
        start_at: str | None = None,
        stop_before: str | None = None,
        allow_all_ce: bool | None = False,
    ) -> None:
        self.image_path = Path(file_path) / "Images"
        self.trans_path = Path(file_path) / "Transforms"
        self.save_path = Path(save_path)
        self.allow_all_ce = allow_all_ce

        self.CE_save_path = self.save_path / "CE"
        self.CE_save_path.mkdir(parents=True, exist_ok=True)
        self.HQ_save_path = self.save_path / "HQ"
        self.HQ_save_path.mkdir(parents=True, exist_ok=True)
        self.LQ_save_path = self.save_path / "LQ"
        self.LQ_save_path.mkdir(parents=True, exist_ok=True)

        self.HU_filter = itk.ClampImageFilter()
        self.HU_filter.SetLowerBound(HU_MIN)
        self.HU_filter.SetUpperBound(HU_MAX)

        try:
            with open(self.save_path / "source_coords.json") as fp:
                self.source_coords = json.load(fp)
        except FileNotFoundError:
            self.source_coords = {}

        if include is None:
            self.subjects = [name for name in os.listdir(self.image_path)]
        else:
            self.subjects = [
                name for name in os.listdir(self.image_path) if name in include
            ]

        self.subjects = sorted(self.subjects)
        idx_start = None if start_at is None else self.subjects.index(start_at)
        idx_end = None if stop_before is None else self.subjects.index(stop_before)
        self.subjects = self.subjects[idx_start:idx_end]

        self.ignore = ignore
        if ignore is not None:
            for subject in ignore["subject_ignore"]:
                if subject in self.subjects:
                    self.subjects.remove(subject)

        self.subjects.sort()

    def _transform_if_required(
        self,
        source_name: str,
        target_name: str,
        source_img: itk.Image,
        target_img: itk.Image,
        subject_path: Path,
    ) -> itk.Image:
        subject_id = subject_path.stem
        transform_path = self.trans_path / subject_id
        transform_candidates = list(
            transform_path.glob(f"{source_name[-3:]}_to_{target_name[-3:]}.h5"),
        )

        source_img = itk.Resample(source_img, target_img, defaultPixelValue=HU_DEFAULT)

        if len(transform_candidates) == 1:
            transform = itk.ReadTransform(str(transform_candidates[0]))
            source_img = itk.Resample(
                source_img,
                transform,
                defaultPixelValue=HU_DEFAULT,
            )

        return source_img

    def _load_images(self, subject_path: Path) -> TypeImageDict | None:
        # Get candidates for CE, HQ non-CE, HQ post-CE, LQ post-CE
        ce_paths = list(subject_path.glob("*AC*.nrrd"))
        hq_paths = list(subject_path.glob("*HQ*.nrrd"))
        lq_paths = list(subject_path.glob("*LQ*.nrrd"))

        if self.ignore is not None:
            ce_paths = [
                p for p in ce_paths if p.stem not in self.ignore["image_ignore"]
            ]
            hq_paths = [
                p for p in hq_paths if p.stem not in self.ignore["image_ignore"]
            ]
            lq_paths = [
                p for p in lq_paths if p.stem not in self.ignore["image_ignore"]
            ]

        nce_path = hq_paths[0]
        hq_paths = hq_paths[1:-1]

        # Ignore CE if multiple
        if len(ce_paths) > 1 and not self.allow_all_ce:
            assert int(ce_paths[0].stem[-3:]) > int(
                nce_path.stem[-3:],
            ), f"{ce_paths[0].stem} vs {nce_path.stem}"
            print(f"{subject_path.stem} CE: {len(ce_paths)}")  # noqa: T201
            return None

        # Ignore CE if comes after needle insertion
        elif len(ce_paths) == 1 and not self.allow_all_ce:
            assert int(ce_paths[0].stem[-3:]) > int(
                nce_path.stem[-3:],
            ), f"{ce_paths[0].stem} vs {nce_path.stem}"

            if int(ce_paths[0].stem[-3:]) > int(hq_paths[0].stem[-3:]):
                print(  # noqa: T201
                    f"{subject_path.stem}"
                    f"CE: {ce_paths[0].stem}, HQ: {hq_paths[0].stem}",
                )
                return None
            else:
                if len(lq_paths) > 0 and int(ce_paths[0].stem[-3:]) > int(
                    lq_paths[0].stem[-3:],
                ):
                    print(  # noqa: T201
                        f"{subject_path.stem}"
                        f"CE: {ce_paths[0].stem}, LQ: {lq_paths[0].stem}",
                    )
                    return None

        # Read images and transform if required
        nce, ace, hqs, lqs = {}, {}, {}, {}
        nce[nce_path.stem] = itk.ReadImage(str(nce_path))

        for img_path in ce_paths:
            ace[img_path.stem] = itk.ReadImage(str(img_path))

        for img_path in hq_paths:
            hqs[img_path.stem] = itk.ReadImage(str(img_path))
            assert hqs[img_path.stem].GetSpacing()[2] == HQ_SLICE_THICK, (
                f"{hqs[img_path.stem]} spacing:" f"{hqs[img_path.stem].GetSpacing()}"
            )
        for img_path in lq_paths:
            lqs[img_path.stem] = img_path

        return nce, ace, hqs, lqs

    def _trim_source(
        self,
        source: itk.Image,
        lq: bool,
    ) -> tuple[itk.Image, int | None, int | None]:
        source_slice_means = itk.GetArrayFromImage(source).mean(axis=(1, 2))
        source_slice_idx = np.argwhere(source_slice_means > HU_THRESHOLD)
        if len(source_slice_idx) == 0 and lq:
            return source, None, None

        source_lower = int(source_slice_idx[0])
        source_upper = int(source_slice_idx[-1]) + 1

        if lq:
            source = source[:, :, source_lower:source_upper:LQ_SLICE_THICK]
            if source.GetDepth() != LQ_DEPTH:
                return source, None, None

        else:
            source = source[:, :, source_lower:source_upper]
            if source.GetDepth() < MIN_HQ_DEPTH:
                return source, None, None

        return source, source_lower, source_upper

    def _save_ce_nce(
        self,
        ace_dict: dict[str, itk.Image],
        nce_dict: dict[str, itk.Image],
        subject_path: Path,
    ) -> None:
        # Process initial non-CE and CE images
        nce_name = list(nce_dict.keys())[0]
        nce = nce_dict[nce_name]

        for ace_name, ace in ace_dict.items():
            ace = self._transform_if_required(
                source_name=ace_name,
                target_name=nce_name,
                source_img=ace,
                target_img=nce,
                subject_path=subject_path,
            )
            ace, ace_lower, ace_upper = self._trim_source(ace, False)
            self.source_coords[ace_name] = {nce_name: [ace_lower, ace_upper]}

            ace = self.HU_filter.Execute(ace)
            nce_npy = itk.GetArrayFromImage(nce).astype("float16")
            ace_npy = itk.GetArrayFromImage(ace).astype("float16")

            np.save(self.CE_save_path / f"{ace_name}.npy", ace_npy)

        # Clamp HU values
        nce = self.HU_filter.Execute(nce)
        nce_npy = itk.GetArrayFromImage(nce).astype("float16")
        np.save(self.HQ_save_path / f"{nce_name}.npy", nce_npy)

    def _save_lq_hq(
        self,
        hqs_dict: dict[str, itk.Image],
        lqs_dict: dict[str, Path],
        nce_dict: dict[str, itk.Image],
        subject_path: Path,
        num_lq: int,
    ) -> None:
        # Process initial non-CE and CE images
        nce_name = list(nce_dict.keys())[0]
        nce = nce_dict[nce_name]

        # Process HQ and LQ post-CE images
        for hq_name, hq in hqs_dict.items():
            hq = self._transform_if_required(
                source_name=hq_name,
                target_name=nce_name,
                source_img=hq,
                target_img=nce,
                subject_path=subject_path,
            )
            hq, hq_lower, hq_upper = self._trim_source(hq, False)
            if hq_lower is None or hq_upper is None:
                print(  # noqa: T201
                    (
                        f"Skipping {hq_name} - no overlap "
                        f"or size wrong {hq.GetDepth()}"
                    ),
                )
                continue
            self.source_coords[hq_name] = {nce_name: [hq_lower, hq_upper]}

            series_no = int(hq_name[-3:])
            lq_candidates = list(lqs_dict.keys())
            lq_candidates = sorted(lqs_dict, key=lambda x: abs(int(x[-3:]) - series_no))
            lq_names = lq_candidates[0:num_lq]
            assert len(lq_names) > 0, f"LQ candidates: {len(lq_names)}"

            for lq_name in lq_names:
                lq = itk.ReadImage(str(lqs_dict[lq_name]))
                if lq.GetSpacing()[2] != LQ_SLICE_THICK:
                    print(f"{lq_name} spacing {lq.GetSpacing()}")  # noqa: T201
                    continue

                hq_candidates = sorted(
                    hqs_dict,
                    key=lambda x: abs(int(x[-3:]) - int(lq_name[-3:])),
                )
                closest_hq = hq_candidates[0]
                lq = self._transform_if_required(
                    source_name=closest_hq,
                    target_name=nce_name,
                    source_img=lq,
                    target_img=nce,
                    subject_path=subject_path,
                )
                lq, lq_lower, lq_upper = self._trim_source(lq, lq=True)
                if lq_lower is None or lq_upper is None:
                    print(  # noqa: T201
                        (
                            f"Skipping {lq_name} - no overlap "
                            f"or size wrong {lq.GetDepth()}"
                        ),
                    )
                    continue

                self.source_coords[lq_name] = {nce_name: [lq_lower, lq_upper]}
                if (
                    lq_lower - hq_lower < 0
                    or lq_lower - hq_lower + MIN_HQ_DEPTH > hq.GetDepth()
                ):
                    print(f"Skipping {lq_name} - extends past {hq_name}")  # noqa: T201
                    continue

                #  Clamp HU values and save
                lq = self.HU_filter.Execute(lq)
                lq_npy = itk.GetArrayFromImage(lq).astype("float16")
                np.save(self.LQ_save_path / f"{lq_name}.npy", lq_npy)

            # Clamp HU values and save
            hq = self.HU_filter.Execute(hq)
            hq_npy = itk.GetArrayFromImage(hq).astype("float16")
            np.save(self.HQ_save_path / f"{hq_name}.npy", hq_npy)

    def process_images(self, num_lq: int = 2) -> None:
        for subject in self.subjects:
            subject_path = self.image_path / subject
            imgs = self._load_images(subject_path)
            if imgs is None:
                continue
            else:
                nce, ace, hqs, lqs = imgs

            if len(ace) > 0:
                self._save_ce_nce(ace, nce, subject_path)

            if len(lqs) > 0:
                self._save_lq_hq(hqs, lqs, nce, subject_path, num_lq)

            self.source_coords = dict(sorted(self.source_coords.items()))
            with open(self.save_path / "source_coords.json", "w") as fp:
                json.dump(self.source_coords, fp, indent=4)

            print(f"{subject} saved")  # noqa: T201

    def check_saved(self) -> None:
        for subject in self.subjects:
            img_paths = list((self.save_path / "CE").glob(f"{subject}*"))
            img_paths += list((self.save_path / "HQ").glob(f"{subject}*"))
            img_paths += list((self.save_path / "LQ").glob(f"{subject}*"))
            if len(img_paths) == 0:
                print(f"No images for {subject}")  # noqa: T201
                continue

            imgs = {}
            for img_path in img_paths:
                imgs[img_path.stem] = np.load(img_path)

            rows = int(np.floor(np.sqrt(len(imgs))))
            cols = int(np.ceil(len(imgs) / rows))

            _, axs = plt.subplots(rows, cols, figsize=(18, 8))
            for i, (img_id, img) in enumerate(imgs.items()):
                axs.ravel()[i].imshow(img[0, :, :], cmap="bone", vmin=-150, vmax=250)
                axs.ravel()[i].axis("off")
                axs.ravel()[i].set_title(img_id)

            plt.show()


# -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-i", type=str, help="Image path")
    parser.add_argument("--save_path", "-s", type=str, help="Save path")
    parser.add_argument("--to_include", "-t", type=str, help="Include IDs")
    parser.add_argument("--start_at", "-sa", type=str, help="Start ID")
    parser.add_argument("--stop_before", "-sb", type=str, help="End ID")
    parser.add_argument(
        "--allow_all_ce",
        "-a",
        help="Allow all CE",
        action="store_true",
    )
    arguments = parser.parse_args()

    if arguments.to_include is not None:
        to_include = arguments.to_include.split(",")
    else:
        to_include = None
    root_dir = Path(__file__).resolve().parents[0]

    with open(root_dir / "ignore.json") as fp:
        ignore = json.load(fp)

    img_conv = ImgConv(
        arguments.image_path,
        arguments.save_path,
        include=to_include,
        ignore=ignore,
        start_at=arguments.start_at,
        stop_before=arguments.stop_before,
        allow_all_ce=arguments.allow_all_ce,
    )
    img_conv.process_images()
    img_conv.check_saved()

    # Check T083A0


# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
