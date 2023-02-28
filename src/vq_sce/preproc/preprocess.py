import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import SimpleITK as itk

from vq_sce import (
    HU_MIN,
    HU_MAX,
    HQ_SLICE_THICK,
    LQ_DEPTH,
    LQ_SLICE_THICK,
    MIN_HQ_DEPTH
)

HU_DEFAULT = -2048
HU_THRESHOLD = -2000


#-------------------------------------------------------------------------

class ImgConv:
    def __init__(
        self,
        file_path: str,
        save_path: str,
        include: list = None,
        ignore: list = None,
        start_at: list = None,
        stop_before: list = None,
        allow_all_ce: bool = False
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
            with open(self.save_path / "source_coords.json", 'r') as fp:
                self.source_coords = json.load(fp)
        except FileNotFoundError:
            self.source_coords = {}

        if include is None:
            self.subjects = [name for name in os.listdir(self.image_path)]
        else:
            self.subjects = [name for name in os.listdir(self.image_path) if name in include]

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
        subject_path: Path
    ) -> itk.Image:

        subject_id = subject_path.stem
        transform_path = (self.trans_path / subject_id)
        transform_candidates = list(transform_path.glob(
            f"{source_name[-3:]}_to_{target_name[-3:]}.h5"
        ))

        source_img = itk.Resample(
            source_img, target_img,
            defaultPixelValue=HU_DEFAULT
        )

        if len(transform_candidates) == 1:
            transform = itk.ReadTransform(str(transform_candidates[0]))
            source_img = itk.Resample(
                source_img, transform,
                defaultPixelValue=HU_DEFAULT
            )

        return source_img

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

        NCE_path = HQ_paths[0]
        HQ_paths = HQ_paths[1:-1]

        # Ignore CE if multiple
        if len(CE_paths) > 1 and not self.allow_all_ce:
            assert int(CE_paths[0].stem[-3:]) > int(NCE_path.stem[-3:]), (
                        f"{CE_paths[0].stem} vs {NCE_path.stem}")
            print(f"{subject_path.stem} CE: {len(CE_paths)}")
            return None

        # Ignore CE if comes after needle insertion
        elif len(CE_paths) == 1 and not self.allow_all_ce:
            assert int(CE_paths[0].stem[-3:]) > int(NCE_path.stem[-3:]), (
                        f"{CE_paths[0].stem} vs {NCE_path.stem}")

            if int(CE_paths[0].stem[-3:]) > int(HQ_paths[0].stem[-3:]):
                print(f"{subject_path.stem} CE: {CE_paths[0].stem}, HQ: {HQ_paths[0].stem}")
                return None
            else:
                if (len(LQ_paths) > 0 and 
                    int(CE_paths[0].stem[-3:]) > int(LQ_paths[0].stem[-3:])):
                    print(f"{subject_path.stem} CE: {CE_paths[0].stem}, LQ: {LQ_paths[0].stem}")
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

    def _trim_source(
        self,
        source: itk.Image,
        lq: bool,
    ) -> tuple[itk.Image, int, int]:

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

    def _save_ce_nce(self, ace: itk.Image, nce: itk.Image, subject_path: Path):

        # Process initial non-CE and CE images
        nce_name = list(nce.keys())[0]
        nce = nce[nce_name]

        for ace_name, ace in ace.items():
            ace = self._transform_if_required(
                source_name=ace_name,
                target_name=nce_name,
                source_img=ace,
                target_img=nce,
                subject_path=subject_path
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
        hqs: itk.Image,
        lqs: itk.Image,
        nce: itk.Image,
        subject_path: Path,
        num_lq: int
    ):
        # Process initial non-CE and CE images
        nce_name = list(nce.keys())[0]
        nce = nce[nce_name]

        # Process HQ and LQ post-CE images
        for hq_name, hq in hqs.items():
            hq = self._transform_if_required(
                source_name=hq_name,
                target_name=nce_name,
                source_img=hq,
                target_img=nce,
                subject_path=subject_path
            )
            hq, hq_lower, hq_upper = self._trim_source(hq, False)
            if hq_lower is None or hq_upper is None:
                print((f"Skipping {hq_name} - no overlap "
                       f"or size wrong {hq.GetDepth()}"))
            self.source_coords[hq_name] = {nce_name: [hq_lower, hq_upper]}

            series_no = int(hq_name[-3:])
            LQ_candidates = list(lqs.keys())
            LQ_candidates = sorted(
                lqs, key=lambda x: abs(int(x[-3:]) - series_no)
            )
            LQ_names = LQ_candidates[0:num_lq]
            assert len(LQ_names) > 0, f"LQ candidates: {len(LQ_names)}"

            for lq_name in LQ_names:
                lq = itk.ReadImage(str(lqs[lq_name]))
                if lq.GetSpacing()[2] != LQ_SLICE_THICK:
                    print(f"{lq_name} spacing {lq.GetSpacing()}")
                    continue

                HQ_candidates = sorted(
                hqs, key=lambda x: abs(int(x[-3:]) - int(lq_name[-3:]))
            )
                closest_hq = HQ_candidates[0]
                lq = self._transform_if_required(
                    source_name=closest_hq,
                    target_name=nce_name,
                    source_img=lq,
                    target_img=nce,
                    subject_path=subject_path
                )
                lq, lq_lower, lq_upper = self._trim_source(lq, lq=True)
                if lq_lower is None or lq_upper is None:
                    print((f"Skipping {lq_name} - no overlap "
                           f"or size wrong {lq.GetDepth()}"))
                    continue

                self.source_coords[lq_name] = {nce_name: [lq_lower, lq_upper]}
                if lq_lower - hq_lower < 0 or lq_lower - hq_lower + MIN_HQ_DEPTH > hq.GetDepth():
                    print(f"Skipping {lq_name} - extends past {hq_name}")
                    continue

                #  Clamp HU values and save
                lq = self.HU_filter.Execute(lq)
                lq_npy = itk.GetArrayFromImage(lq).astype("float16")
                np.save(self.LQ_save_path / f"{lq_name}.npy", lq_npy)

            # Clamp HU values and save
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

            if len(ace) > 0:
                self._save_ce_nce(ace, nce, subject_path)

            if len(LQs) > 0:
                self._save_lq_hq(HQs, LQs, nce, subject_path, num_LQ)

            self.source_coords = dict(sorted(self.source_coords.items()))
            with open(self.save_path / "source_coords.json", 'w') as fp:
                json.dump(self.source_coords, fp, indent=4)

            print(f"{subject} saved")

    def check_saved(self):

        for subject in self.subjects:
            img_paths = list((self.save_path / "CE").glob(f"{subject}*"))
            img_paths += list((self.save_path / "HQ").glob(f"{subject}*"))
            img_paths += list((self.save_path / "LQ").glob(f"{subject}*"))
            if len(img_paths) == 0:
                print(f"No images for {subject}")
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


#-------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", '-i', type=str, help="Image path")
    parser.add_argument("--save_path", '-s', type=str, help="Save path")
    parser.add_argument("--to_include", '-t', type=str, help="Include IDs")
    parser.add_argument("--start_at", '-sa', type=str, help="Start ID")
    parser.add_argument("--stop_before", '-sb', type=str, help="End ID")
    parser.add_argument("--allow_all_ce", '-a', help="Allow all CE", action="store_true")
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
        stop_before=arguments.stop_before,
        allow_all_ce=arguments.allow_all_ce
    )
    img_conv.process_images()
    img_conv.check_saved()

    # Check T083A0

#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
