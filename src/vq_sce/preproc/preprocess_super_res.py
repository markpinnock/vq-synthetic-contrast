import json
import numpy as np
import os
from pathlib import Path
import SimpleITK as itk

HU_MIN = -500
HU_MAX = 2500


class ImgConvSuperRes:
    def __init__(
        self,
        file_path: str,
        save_path: str,
        include: list = None,
        ignore: list = None
    ) -> None:

        self.image_path = Path(file_path) / "Images"
        self.trans_path = Path(file_path) / "Transformations"
        self.save_path = Path(save_path)
        self.HQ_save_path = self.save_path / "HQ"
        self.HQ_save_path.mkdir(parents=True, exist_ok=True)
        self.LQ_save_path = self.save_path / "LQ"
        self.LQ_save_path.mkdir(parents=True, exist_ok=True)

        self.HU_min = -2048
        self.abdo_window_min = -150
        self.abdo_window_max = 250
        self.HU_min_all = 2048
        self.HU_max_all = -2048

        self.HU_filter = itk.ClampImageFilter()
        self.HU_filter.SetLowerBound(HU_MIN)
        self.HU_filter.SetUpperBound(HU_MAX)

        if include is None:
            self.subjects = [name for name in os.listdir(self.image_path) if 'F' not in name]
        else:
            self.subjects = [name for name in os.listdir(self.image_path) if name in include and 'F' not in name]

        if ignore is not None:
            for subject in ignore:
                if subject in self.subjects:
                    self.subjects.remove(subject)

        self.subjects.sort()

    def _load_images(
        self,
        subject_path: Path
    ) -> tuple[dict[str, itk.Image], dict[str, Path]] | None:

        HQ_paths = list(subject_path.glob("*HQ*.nrrd"))
        LQ_paths = list(subject_path.glob("*LQ*.nrrd"))
        if len(HQ_paths) == 0 or len(LQ_paths) == 0:
            return None

        HQs, LQs = {}, {}
        for img_path in HQ_paths[1:-1]:
            HQs[img_path.stem] = itk.ReadImage(str(img_path))
            assert HQs[img_path.stem].GetSpacing()[2] == 1.0, (
                f"{HQs[img_path.stem]} spacing:"
                f"{HQs[img_path.stem].GetSpacing()}"
            )
        for img_path in LQ_paths:
            LQs[img_path.stem] = img_path

        return HQs, LQs

    def process_images(self, num_LQ: int = 2):
        relative_origins = {}
        for subject in self.subjects:
            subject_path = self.image_path / subject
            HQLQ = self._load_images(subject_path)
            if HQLQ is None:
                continue
            else:
                HQs, LQs = HQLQ

            for HQ_name, HQ_img in HQs.items():
                HQ_img = self.HU_filter.Execute(HQ_img)
                series_no = int(HQ_name[-3:])
                LQ_candidates = list(LQs.keys())
                LQ_candidates = sorted(
                    LQs,
                    key=lambda x: abs(int(x[-3:]) - series_no)
                )
                LQ_names = LQ_candidates[0:num_LQ]
                assert len(LQ_names) > 0, f"LQ candidates: {len(LQ_names)}"
                for LQ_name in LQ_names:
                    LQ_img = itk.ReadImage(str(LQs[LQ_name]))
                    if LQ_img.GetSpacing()[2] != 4.0:
                        continue
                    LQ_rel_origin = int(np.round(LQ_img.GetOrigin()[2] - HQ_img.GetOrigin()[2]))
                    if LQ_rel_origin < 0:
                        continue
                    LQ_img = self.HU_filter.Execute(LQ_img)

                    # Check image is orientated correctly and flip/rotate if necessary
                    image_dir = np.around(LQ_img.GetDirection())
                    if image_dir[0] == 0.0 or image_dir[4] == 0.0:
                        LQ_img = itk.PermuteAxes(LQ_img, [1, 0, 2])
                        image_dir = np.around(LQ_img.GetDirection())
                        LQ_img = LQ_img[::int(image_dir[0]), ::int(image_dir[4]), :]

                    else:
                        LQ_img = LQ_img[::int(image_dir[0]), ::int(image_dir[4]), :]

                    relative_origins[LQ_name] = [HQ_name, LQ_rel_origin]
                    LQ_npy = itk.GetArrayFromImage(LQ_img).transpose([1, 2, 0]).astype("float16")
                    np.save(self.LQ_save_path / f"{LQ_name}.npy", LQ_npy)
                    print(f"{LQ_name} saved")

                # Check image is orientated correctly and flip/rotate if necessary
                image_dir = np.around(HQ_img.GetDirection())
                if image_dir[0] == 0.0 or image_dir[4] == 0.0:
                    HQ_img = itk.PermuteAxes(HQ_img, [1, 0, 2])
                    image_dir = np.around(HQ_img.GetDirection())
                    HQ_img = HQ_img[::int(image_dir[0]), ::int(image_dir[4]), :]

                else:
                    HQ_img = HQ_img[::int(image_dir[0]), ::int(image_dir[4]), :]                

                if LQ_rel_origin < 0:
                    continue

                HQ_npy = itk.GetArrayFromImage(HQ_img).transpose([1, 2, 0]).astype("float16")
                np.save(self.HQ_save_path / f"{HQ_name}.npy", HQ_npy)
                print(f"{HQ_name} saved")
                with open(self.save_path / "relative_origins.json", 'w') as fp:
                    json.dump(relative_origins, fp, indent=4)


if __name__ == "__main__":
    img_path = "Z:/Clean_CT_Data/Toshiba"
    save_path = "D:/ProjectImages/SuperRes"

    with open("vec_quant_sCE/preproc/ignore.json", 'r') as fp:
        ignore = json.load(fp)

    subject_ignore = list(ignore["subject_ignore"].keys())
    image_ignore = ignore["image_ignore"]
    to_include = ["T044A0", "T048A0", "T049A0", "T050A0"]

    img_conv = ImgConvSuperRes(
        img_path,
        save_path,
        include=to_include,
        ignore=subject_ignore
    )
    img_conv.process_images()
