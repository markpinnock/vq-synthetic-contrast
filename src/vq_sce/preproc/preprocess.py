import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import SimpleITK as itk

from vq_sce import HU_MIN, HU_MAX
from . import (
    HQ_SLICE_THICK,
    LQ_DEPTH,
    LQ_SLICE_THICK
)


#-------------------------------------------------------------------------

class ImgConv:
    def __init__(
        self,
        file_path: str,
        save_path: str,
        include: list = None,
        ignore: list = None,
        start_at: list = None,
        stop_before: list = None
    ) -> None:

        self.image_path = Path(file_path) / "Images"
        self.trans_path = Path(file_path) / "Transforms"
        self.save_path = Path(save_path)

        self.CE_save_path = self.save_path / "CE"
        self.CE_save_path.mkdir(parents=True, exist_ok=True)
        self.HQ_save_path = self.save_path / "HQ"
        self.HQ_save_path.mkdir(parents=True, exist_ok=True)
        self.LQ_save_path = self.save_path / "LQ"
        self.LQ_save_path.mkdir(parents=True, exist_ok=True)
        self.HU_min = -2048

        self.HU_filter = itk.ClampImageFilter()
        self.HU_filter.SetLowerBound(HU_MIN)
        self.HU_filter.SetUpperBound(HU_MAX)

        try:
            with open(self.save_path / "source_coords.json", 'r') as fp:
                self.source_coords = json.load(fp)
        except FileNotFoundError:
            self.source_coords = {}

        if include is None:
            self.subjects = [name for name in os.listdir(self.image_path)]
        else:
            self.subjects = [name for name in os.listdir(self.image_path) if name in include]

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
        subject_path: Path
    ) -> itk.Image:

        subject_id = subject_path.stem
        transform_path = (self.trans_path / subject_id)
        transform_candidates = list(transform_path.glob(
            f"{source_name[-3:]}_to_{target_name[-3:]}.h5"
        ))

        source_img = itk.Resample(
            source_img, target_img,
            defaultPixelValue=self.HU_min
        )

        if len(transform_candidates) == 1:
            transform = itk.ReadTransform(str(transform_candidates[0]))
            source_img = itk.Resample(
                source_img, transform,
                defaultPixelValue=self.HU_min
            )
            z_shift = transform.GetParameters()[-1]
        else:
            z_shift = 0

        return source_img, round(z_shift)

    def _load_images(
        self,
        subject_path: Path
    ) -> tuple[dict[str, itk.Image | Path]] | None:

        # Get candidates for CE, HQ non-CE, HQ post-CE, LQ post-CE
        CE_paths = list(subject_path.glob("*AC*.nrrd"))
        HQ_paths = list(subject_path.glob("*HQ*.nrrd"))
        LQ_paths = list(subject_path.glob("*LQ*.nrrd"))
        
        if self.ignore is not None:
            CE_paths = [
                p for p in CE_paths if p.stem not in self.ignore["image_ignore"]
            ]
            HQ_paths = [
                p for p in HQ_paths if p.stem not in self.ignore["image_ignore"]
            ]
            LQ_paths = [
                p for p in LQ_paths if p.stem not in self.ignore["image_ignore"]
            ]

        if len(CE_paths) > 1:
            print(f"{subject_path.stem} CE: {len(CE_paths)}")
            return None

        NCE_path = HQ_paths[0]
        HQ_paths = HQ_paths[1:-1]

        # Ensure non-CE scan is before CE
        if len(CE_paths) == 1:
            assert int(CE_paths[0].stem[-3:]) > int(NCE_path.stem[-3:]), (
                        f"{CE_paths[0].stem} vs {NCE_path.stem}")

            if (int(CE_paths[0].stem[-3:]) > int(HQ_paths[0].stem[-3:]) or
                int(CE_paths[0].stem[-3:]) > int(LQ_paths[0].stem[-3:])):
                print(f"{subject_path.stem} CE: {CE_paths[0].stem}, HQ: {HQ_paths[0].stem}")
                return None

        # Read images and transform if required
        NCE, ACE, HQs, LQs = {}, {}, {}, {}
        NCE[NCE_path.stem] = itk.ReadImage(str(NCE_path))

        for img_path in CE_paths:
            ACE[img_path.stem] = itk.ReadImage(str(img_path))

        for img_path in HQ_paths:
            HQs[img_path.stem] = itk.ReadImage(str(img_path))
            assert HQs[img_path.stem].GetSpacing()[2] == HQ_SLICE_THICK, (
                f"{HQs[img_path.stem]} spacing:"
                f"{HQs[img_path.stem].GetSpacing()}"
            )
        for img_path in LQ_paths:
            LQs[img_path.stem] = img_path

        return NCE, ACE, HQs, LQs

    # def _check_orientation(self, img: itk.Image) -> itk.Image:
    #     """ Check image is orientated correctly
    #         and flip/rotate if necessary
    #     """

    #     image_dir = np.around(img.GetDirection())
    #     if image_dir[0] == 0.0 or image_dir[4] == 0.0:
    #         img = itk.PermuteAxes(img, [1, 0, 2])
    #         image_dir = np.around(img.GetDirection())
    #         img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

    #     else:
    #         img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

    #     return img

    def _get_source_coords(
        self,
        source: itk.Image,
        target: itk.Image
    ) -> tuple[int, int]:

        source_lower = int(np.round(source.GetOrigin()[2]
                            - target.GetOrigin()[2]))
        source_upper = source_lower + source.GetDepth()

        return source_lower, source_upper

    def _trim_source(
        self,
        source: itk.Image,
        target: itk.Image,
        source_lower: int,
        source_upper: int
    ) -> tuple[itk.Image, int, int]:

        target_z = target.GetDepth()

        if source_lower > 0 and source_upper > target_z:
            source_upper = target_z
        elif source_lower < 0 and source_upper < target_z:
            source_lower = 0
        elif source_lower > 0 and source_upper < target_z:
            pass
        elif source_lower < 0 and source_upper > target_z:
            source_lower = 0
            source_upper = target_z

        source = source[:, :, source_lower:source_upper]

        return source, source_lower, source_upper

    def _save_ce_nce(self, ace: itk.Image, nce: itk.Image, subject_path: Path):

            # Process initial non-CE and CE images
            nce_name = list(nce.keys())[0]
            ace_name = list(ace.keys())[0]
            nce = nce[nce_name]
            ace = ace[ace_name]

            # Check image is orientated correctly
            nce = self._check_orientation(nce)
            ace = self._check_orientation(ace)
            ace_lower, ace_upper = self._get_source_coords(ace, nce)
            ace, z_shift = self._transform_if_required(
                source_name=ace_name,
                target_name=nce_name,
                source_img=ace,
                target_img=nce,
                subject_path=subject_path
            )
            ace_trim, ace_lower, ace_upper = self._trim_source(
                source=ace,
                target=nce,
                source_lower=ace_lower - z_shift,
                source_upper=ace_upper - z_shift
            )

            # Occasionally need to adjust bounds and re-trim
            ace_npy = itk.GetArrayFromImage(ace_trim).astype("float16")
            if (ace_npy[0, :, :].mean() == self.HU_min
                and ace_npy[-1, :, :].mean() == self.HU_min):
                raise ValueError(
                    f"{ace_name}: {ace_npy[0, :, :].mean()}, {ace_npy[-1, :, :].mean()}"
                )

            elif ace_npy[0, :, :].mean() == self.HU_min:
                ace_lower += 1
                ace_upper += 1
                ace_trim, ace_lower, ace_upper = self._trim_source(
                    source=ace,
                    target=nce,
                    source_lower=ace_lower,
                    source_upper=ace_upper
                )

            elif ace_npy[-1, :, :].mean() == self.HU_min:
                ace_lower -= 1
                ace_upper -= 1
                ace_trim, ace_lower, ace_upper = self._trim_source(
                    source=ace,
                    target=nce,
                    source_lower=ace_lower,
                    source_upper=ace_upper
                )

            else:
                pass

            self.source_coords[ace_name] = {nce_name: [ace_lower, ace_upper]}

            # Clamp HU values
            nce = self.HU_filter.Execute(nce)
            ace = self.HU_filter.Execute(ace)

            nce_npy = itk.GetArrayFromImage(nce).astype("float16")
            ace_npy = itk.GetArrayFromImage(ace_trim).astype("float16")

            np.save(self.HQ_save_path / f"{nce_name}.npy", nce_npy)
            np.save(self.CE_save_path / f"{ace_name}.npy", ace_npy)

    def _save_lq_hq(
        self,
        hqs: itk.Image,
        lqs: itk.Image,
        subject_path: Path,
        num_lq: int
    ):
        # Process HQ and LQ post-CE images
        for hq_name, hq in hqs.items():

            # Check image is orientated correctly
            hq = self._check_orientation(hq)

            series_no = int(hq_name[-3:])
            LQ_candidates = list(lqs.keys())
            LQ_candidates = sorted(
                lqs, key=lambda x: abs(int(x[-3:]) - series_no)
            )
            LQ_names = LQ_candidates[0:num_lq]
            assert len(LQ_names) > 0, f"LQ candidates: {len(LQ_names)}"

            for lq_name in LQ_names:
                if lq_name not in self.source_coords.keys():
                    self.source_coords[lq_name] = {}

                lq = itk.ReadImage(str(lqs[lq_name]))
                if lq.GetSpacing()[2] != LQ_SLICE_THICK:
                    continue

                lq_lower, lq_upper = self._get_source_coords(lq, hq)
                if lq_lower < 0:
                    continue

                # Check image is orientated correctly and clamp values
                lq = self._check_orientation(lq)
                lq, _ = self._transform_if_required(
                    source_name=lq_name,
                    target_name=hq_name,
                    source_img=lq,
                    target_img=hq,
                    subject_path=subject_path
                )
                lq, lq_lower, lq_upper = self._trim_source(
                    source=lq,
                    target=hq,
                    source_lower=lq_lower,
                    source_upper=lq_upper
                )
                lq_upper = lq_lower + LQ_DEPTH * LQ_SLICE_THICK

                self.source_coords[lq_name][hq_name] = [lq_lower, lq_upper]

                #  Clamp HU values
                lq = self.HU_filter.Execute(lq)
                lq_npy = itk.GetArrayFromImage(lq).astype("float16")
                if (lq_npy[0, :, :].mean() == HU_MIN
                    or lq_npy[-1, :, :].mean() == HU_MIN):
                    raise ValueError(
                        f"{lq_name}: {lq_npy[0, :, :].mean()}, {lq_npy[-1, :, :].mean()}"
                    )
                np.save(self.LQ_save_path / f"{lq_name}.npy", lq_npy)

            # Clamp HU values
            hq = self.HU_filter.Execute(hq)
            hq_npy = itk.GetArrayFromImage(hq).astype("float16")
            np.save(self.HQ_save_path / f"{hq_name}.npy", hq_npy)

    def process_images(self, num_LQ: int = 2):

        for subject in self.subjects:
            subject_path = self.image_path / subject
            imgs = self._load_images(subject_path)
            if imgs is None:
                continue
            else:
                nce, ace, HQs, LQs = imgs

            if len(ace) == 1:
                self._save_ce_nce(ace, nce, subject_path)

            # TODO elif

            if len(LQs) > 0:
                self._save_lq_hq(HQs, LQs, subject_path, num_LQ)

            self.source_coords = dict(sorted(self.source_coords.items()))
            with open(self.save_path / "source_coords.json", 'w') as fp:
                json.dump(self.source_coords, fp, indent=4)

            print(f"{subject} saved")


    @staticmethod
    def check_saved(save_path: str, include: list | None = None):

        save_path = Path(save_path)
        with open(save_path / "source_coords.json", 'r') as fp:
            source_coords = json.load(fp)

        if include is not None:
            subjects = include
        else:
            subjects = []
            for img_id in source_coords.keys():
                if img_id[0:6] not in subjects:
                    subjects.append(img_id[0:6])

        for subject in subjects:
            img_paths = list((save_path / "CE").glob(f"{subject}*"))
            img_paths += list((save_path / "HQ").glob(f"{subject}*"))
            img_paths += list((save_path / "LQ").glob(f"{subject}*"))

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


#-------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", '-i', type=str, help="Image path")
    parser.add_argument("--save_path", '-s', type=str, help="Save path")
    parser.add_argument("--to_include", '-t', type=str, help="Include IDs")
    parser.add_argument("--start_at", '-sa', type=str, help="Start ID")
    parser.add_argument("--stop_before", '-sb', type=str, help="End ID")
    arguments = parser.parse_args()

    if arguments.to_include is not None:
        to_include = arguments.to_include.split(',')
    else:
        to_include = None
    root_dir = Path(__file__).resolve().parents[0]

    with open(root_dir / "ignore.json", 'r') as fp:
        ignore = json.load(fp)

    img_conv = ImgConv(
        arguments.image_path,
        arguments.save_path,
        include=to_include,
        ignore=ignore,
        start_at=arguments.start_at,
        stop_before=arguments.stop_before
    )
    img_conv.process_images()
    img_conv.check_saved(arguments.save_path, to_include)

    # Check T025A1, T161A1, T029A0AC007, T071A0, T083A0, T104A0, T114A0, T121A0AC005, T123A1, T150A0
    # T065A1, T066A0, T069A0, T086A0, T088A0, T107A0, T115A0, T126A0, T136A0


#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
