import abc
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as itk


"""
>>> def load(directory):
...     ACs = [f for f in os.listdir(directory) if 'AC' in f]
...     assert len(ACs) == 1
...     HQs = [f for f in os.listdir(directory) if 'HQ' in f]
...     AC = sitk.ReadImage(f"{directory}/{ACs[0]}")
...     HQ = {HQs[i]: sitk.ReadImage(f"{directory}/{HQs[i]}") for i in range(len(HQs))}
...     return AC, HQ

>>> def print_data(directory):
...     AC, HQ = load(directory)
...     print("==================")
...     print(AC.GetDepth(), AC.GetDirection(), AC.GetMetaData("NRRD_space"), AC.GetOrigin(), AC.GetSpacing())
...     print("\n")
...     for n, f in HQ.items():
...         print(n)
...         print(f.GetDepth(), f.GetDirection(), f.GetMetaData("NRRD_space"), f.GetOrigin(), f.GetSpacing())

"""

class ImgConvBase(abc.ABC):

    def __init__(self, image_path, segmentation_path, transformation_path, save_path, output_dims, include, ignore, NCC_tol):
        self.img_path = image_path
        self.seg_path = segmentation_path
        self.trans_path = transformation_path
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_dims = output_dims
        self.NCC_tol = NCC_tol
        self.HU_min = -2048
        self.abdo_window_min = -150
        self.abdo_window_max = 250
        self.HU_min_all = 2048
        self.HU_max_all = -2048

        if include is None:
            self.subjects = [name for name in os.listdir(self.img_path) if 'F' not in name and name not in ignore]
        else:
            self.subjects = [name for name in os.listdir(self.img_path) if name in include and 'F' not in name and name not in ignore]

        self.subjects.sort()

    def list_images(self, ignore: list = None, num_HQ: int = 1000, num_AC: int = 1000, num_VC: int = 1000):
        self.num_HQ = num_HQ
        self.num_AC = num_AC
        self.num_VC = num_VC
        self.HQ_names = {}
        self.AC_names = {}
        self.VC_names = {}

        for name in self.subjects:
            img_path = f"{self.img_path}/{name}/"
            imgs = os.listdir(img_path)
            imgs.sort()

            self.HQ_names[name] = []
            self.AC_names[name] = []
            self.VC_names[name] = []

            for im in imgs:
                if im[-16:-5] not in ignore:
                    if 'HQ' in im:
                        self.HQ_names[name].append(f"{img_path}{im}")
                    elif 'AC' in im:
                        self.AC_names[name].append(f"{img_path}{im}")
                    elif 'VC' in im:
                        self.VC_names[name].append(f"{img_path}{im}")
                    else:
                        continue

            self.HQ_names[name] = self.HQ_names[name][0:num_HQ]
            self.AC_names[name] = self.AC_names[name][0:num_AC]
            self.VC_names[name] = self.VC_names[name][0:num_VC]

            assert len(self.AC_names[name]) == 1
            assert len(self.VC_names[name]) == 1

        return self
    
    def load_subject(self, subject_ID: str, HU_min: int = None, HU_max: int = None) -> list:
        image_names = self.AC_names[subject_ID] + self.VC_names[subject_ID] + self.HQ_names[subject_ID]
        assert len(image_names) > 1
        images = []
        seg_names = []
        segs = []

        for i in range(len(image_names)):
            transform_candidates = glob.glob(f"{self.trans_path}/{subject_ID}/{image_names[i][-8:-5]}_to_*")

            if len(transform_candidates) == 1:
                transform = itk.ReadTransform(transform_candidates[0])
            elif len(transform_candidates) == 0:
                transform = None
            else:
                raise ValueError(transform_candidates)

            img = itk.ReadImage(image_names[i])

            if transform is not None:
                img = itk.Resample(img, transform, defaultPixelValue=self.HU_min)

            image_dir = np.around(img.GetDirection())
            assert img.GetSpacing()[2] == 1.0, f"{image_names[i]}: {img.GetSpacing()}"

            if self.seg_path is not None:
                seg_name = f"{self.seg_path}/{subject_ID}/{image_names[i][-16:-5]}-label.nrrd"

                try:
                    seg = itk.ReadImage(seg_name)

                    if transform is not None:
                        seg = itk.Resample(seg, transform, defaultPixelValue=0)

                except RuntimeError:
                    print(f"Segmentation not found for {image_names[i][-16:-5]}")
                    seg = None
                
                else:
                    seg_names.append(seg_name)
                    assert np.isclose(img.GetSpacing(), seg.GetSpacing()).all() and np.isclose(img.GetDirection(), seg.GetDirection()).all(), f"{image_names[i]}: {img.GetSpacing()}, {seg.GetSpacing()}, {img.GetDirection()}, {seg.GetDirection()}"
                    assert np.isclose(img.GetOrigin(), seg.GetOrigin()).all(), f"{image_names[i]}: {img.GetOrigin()}, {seg.GetOrigin()}"

            # Check image is orientated correctly and flip/rotate if necessary
            if image_dir[0] == 0.0 or image_dir[4] == 0.0:
                img = itk.PermuteAxes(img, [1, 0, 2])
                image_dir = np.around(img.GetDirection())
                img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

                if seg is not None:
                    seg = itk.PermuteAxes(seg, [1, 0, 2])
                    seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                    segs.append(seg)

            else:
                img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

                if seg is not None:
                    seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                    segs.append(seg)

            images.append(img)

        image_bounds = []

        # Get start/end coords data for images
        for img in images:
            image_dim_z = img.GetSize()[2]
            image_origin_z = img.GetOrigin()[2]

            # TODO: Will this work without rounding?
            image_bounds.append(np.around([image_origin_z, image_origin_z + image_dim_z - 1]).astype(np.int32))

        image_bounds = np.vstack(image_bounds)
        tightest_bounds = [image_bounds[:, 0].max(), image_bounds[:, 1].min()]

        for i in range(len(images)):
            start = tightest_bounds[0] - image_bounds[i, 0]
            end = tightest_bounds[1] - image_bounds[i, 1]

            if end == 0:
                images[i] = images[i][:, :, start:]
            else:
                images[i] = images[i][:, :, start:end]

            if images[i].GetSize()[2] == 0:
                print(f"{subject_ID} img: size {images[i].GetSize()}")
                return None

        for i in range(len(segs)):
            # TODO: Will this work without rounding?
            seg_bounds = np.around([segs[i].GetOrigin()[2], segs[i].GetOrigin()[2] + segs[i].GetSize()[2] - 1]).astype(np.int32)
            start = tightest_bounds[0] - seg_bounds[0]
            end = tightest_bounds[1] - seg_bounds[1]

            if end == 0:
                segs[i] = segs[i][:, :, start:]
            else:
                segs[i] = segs[i][:, :, start:end]

            if segs[i].GetSize()[2] == 0:
                print(f"{subject_ID} seg: size {segs[i].GetSize()}")
                return None

        if HU_min is not None and HU_max is not None:
            self.HU_min_all = HU_min
            self.HU_max_all = HU_max
            filter = itk.ClampImageFilter()
            filter.SetLowerBound(HU_min)
            filter.SetUpperBound(HU_max)

        # TODO: allow choosing which image to resample to
        # Resample source to target coords
        for i in range(1, len(images)):
            images[i] = itk.Resample(images[i], images[0], defaultPixelValue=self.HU_min)
            assert images[i].GetSize() == images[0].GetSize()

            # Clip target to window if needed
            if HU_min is not None and HU_max is not None:
                images[i] = filter.Execute(images[i])
            else:
                self.HU_min_all = np.min([self.HU_min_all, itk.GetArrayFromImage(images[i]).min()])
                self.HU_max_all = np.max([self.HU_max_all, itk.GetArrayFromImage(images[i]).max()])

        if HU_min is not None and HU_max is not None:
            images[0] = filter.Execute(images[0])

        # TODO: allow choosing which image to resample to
        for i in range(len(segs)):
            segs[i] = itk.Resample(segs[i], images[0], defaultPixelValue=0)
            assert segs[i].GetSize() == images[0].GetSize()

        assert len(image_names) == len(images)

        if len(segs) > 0:
            return {name: img for name, img in zip(image_names, images)}, {name: seg for name, seg in zip(seg_names, segs)}
        else:
            return {name: img for name, img in zip(image_names, images)}
    
    # TODO: enable ability to select reference image
    def display(self, subject_ID: list = None, display=True, HU_min=None, HU_max=None):
        if subject_ID:
            subjects = subject_ID
        else:
            subjects = self.subjects
        
        for subject in subjects:
            proc = self.load_subject(subject, HU_min=HU_min, HU_max=HU_max)

            if isinstance(proc, tuple):
                imgs = proc[0]
                segs = proc[1]

            elif proc == None:
                print(f"{subject} skipped")
                continue
            
            elif isinstance(proc, dict):
                imgs = proc
                segs = None
            
            else:
                raise ValueError(type(proc))

            img_arrays = [itk.GetArrayFromImage(i).transpose([1, 2, 0]) for i in imgs.values()]
            img_names = list(imgs.keys())
            NCCs = [self.calc_NCC(img_arrays[i], img_arrays[0]) for i in range(len(img_arrays))]
            print(subject, img_arrays[0].shape)

            if segs is not None:
                seg_arrays = [itk.GetArrayFromImage(s).transpose([1, 2, 0]) for s in segs.values()]

            else:
                seg_arrays = []

            if display:
                mid_image = img_arrays[0].shape[2] // 2

                if segs is not None:
                    fig, axs = plt.subplots(4, len(img_arrays))
                else:
                    fig, axs = plt.subplots(3, len(img_arrays))

                for i in range(len(img_arrays)):
                    axs[0, i].imshow(img_arrays[i][:, :, mid_image], cmap="gray")
                    axs[0, i].set_title(img_names[i][-16:])
                    axs[0, i].axis("off")
                    axs[1, i].hist(img_arrays[i].ravel(), bins=100)
                    axs[2, i].imshow(img_arrays[0][:, :, mid_image] - img_arrays[i][:, :, mid_image], cmap="gray")
                    axs[2, i].axis("off")
                    axs[2, i].set_title(f"{NCCs[i]:.4f}")
                
                    if segs is not None and len(seg_arrays) > i:
                        axs[3, i].imshow(seg_arrays[i][:, :, mid_image], cmap="gray")
                        axs[3, i].axis("off")

                plt.show()
                    # plt.pause(0.5)
                    # plt.cla()

    def normalise_image(self, im: np.array) -> np.array:
        return (im - self.HU_min_all) / (self.HU_max_all - self.HU_min_all)

    def calc_NCC(self, a: np.array, b: np.array) -> float:
        assert len(a.shape) == 3
        a = self.normalise_image(a)
        b = self.normalise_image(b)
        N = np.prod(a.shape)

        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sig_a = np.std(a)
        sig_b = np.std(b)

        return np.sum((a - mu_a) * (b - mu_b) / (N * sig_a * sig_b))

    def calc_RBF(self, a: np.array, b: np.array, gamma: float) -> float:
        a = self.normalise_image(a)
        b = self.normalise_image(b)

        return np.exp(-gamma * np.sum(np.power(a - b, 2)))

    def segmentation_com(self, seg):
        seg_X, seg_Y = np.meshgrid(np.linspace(0, self.output_dims[0] - 1, self.output_dims[0]), np.linspace(0, self.output_dims[1] - 1, output_dims[1]))

        if not seg[:, :, 5].sum():
            return [seg.shape[0] // 2, seg.shape[0] // 2]

        x_coord = seg_X[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)
        y_coord = seg_Y[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)

        return [x_coord, y_coord]

    @abc.abstractmethod
    def save_data(self, subject_ID: list = None, HU_min=None, HU_max=None) -> int:
        raise NotImplementedError

    @staticmethod
    def check_seg_dims(img_path: str, seg_path: str, phase: str = 'AC'):
        for f in os.listdir(img_path):
            if f"{f}S" not in os.listdir(seg_path):
                print(f"No segmentations matching folder {f}")
                continue

            ims = [i for i in os.listdir(f"{img_path}/{f}") if phase in i]
            
            if len(ims) == 0:
                print(f"No images matching phase {phase} in folder {f}")
                continue
            
            for im in ims:
                img = itk.ReadImage(f"{img_path}/{f}/{im}")

                try:
                    seg = itk.ReadImage(f"{seg_path}/{f}/{im[:-5]}-label.nrrd")
                except FileNotFoundError:
                    print(f"No segmentation matching {im}")
                else:
                    print(im, img.GetSize(), seg.GetSize())
                    if (np.array(img.GetSize()) != np.array(seg.GetSize())).all(): print(f"Mismatched dims for {im}")
    
    @abc.abstractmethod
    def check_processed_imgs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def check_saved(self):
        raise NotImplementedError


class Paired(ImgConvBase):

    def __init__(self, image_path=None, segmentation_path=None, transformation_path=None, save_path=None, output_dims=None, ignore=[], NCC_tol=None):
        super().__init__(image_path, segmentation_path, transformation_path, save_path, output_dims, ignore, NCC_tol)
    
    def save_data(self, subject_ID: list = None, HU_min=None, HU_max=None) -> int:
        if subject_ID:
            subjects = subject_ID
        else:
            subjects = self.subjects
        
        count = 1
        total = len(subjects)
        
        for subject in subjects:
            proc = self.load_subject(subject, HU_min=HU_min, HU_max=HU_max)

            if isinstance(proc, tuple):
                imgs = proc[0]
                segs = proc[1]

            elif proc == None:
                print(f"{subject} skipped")
                continue
            
            elif isinstance(proc, dict):
                imgs = proc
                segs = None
            
            else:
                raise ValueError(type(proc))

            img_arrays = [itk.GetArrayFromImage(i).transpose([1, 2, 0]).astype("float16") for i in imgs.values()]
            img_names = list(imgs.keys())

            if segs is not None:
                seg_arrays = [itk.GetArrayFromImage(s).transpose([1, 2, 0]).astype("int8") for s in segs.values()]
                seg_names = list(segs.keys())

            vol_thick = img_arrays[0].shape[2]      

            # Partition into sub-volumes
            for name, img in zip(img_names, img_arrays):
                idx = 0  
                stem = name[-16:-5]

                if not os.path.exists(f"{self.save_path}/Images/{stem[6:8]}"):
                    os.makedirs(f"{self.save_path}/Images/{stem[6:8]}")

                for i in range(0, vol_thick, self.output_dims[2]):
                    if i + self.output_dims[2] > vol_thick:
                        break

                    sub_vol = img[:, :, i:i + self.output_dims[2]]
                    np.save(f"{self.save_path}/Images/{stem[6:8]}/{stem}_{idx:03d}.npy", sub_vol)
                    idx += 1
                    count += 1

            for name, seg in zip(seg_names, seg_arrays):
                idx = 0  
                stem = name[-20:-9]

                if not os.path.exists(f"{self.save_path}/Segmentations/{stem[6:8]}"):
                    os.makedirs(f"{self.save_path}/Segmentations/{stem[6:8]}")

                for i in range(0, vol_thick, self.output_dims[2]):
                    if i + self.output_dims[2] > vol_thick:
                        break

                    sub_vol = seg[:, :, i:i + self.output_dims[2]]
                    np.save(f"{self.save_path}/Segmentations/{stem[6:8]}/{stem}_{idx:03d}.npy", sub_vol)
                    idx += 1
                    count += 1

        return count

    @staticmethod
    def check_saved(self):
        subjects = []
        phases = os.listdir(self.save_path)

        for img in os.listdir(f"{self.save_path}/Images/{phases[0]}"):
            if img[:6] not in subjects: subjects.append(img[:6])
        
        for img_name in subjects:
            subject_imgs = []

            for phase in phases:
                path = f"{self.save_path}/Images/{phase}"
                imgs = os.listdir(path)
                subject_imgs += [np.load(f"{path}/{im}") for im in imgs if img_name in im]
                path = f"{self.save_path}/Segmentations/{phase}"
                segs = os.listdir(path)
                subject_imgs += [np.load(f"{path}/{im}") for im in segs if img_name in im]

            if len(imgs) % 2 == 0:
                rows = 2
                cols = len(imgs) // rows
            else:
                rows = 2
                cols = len(imgs) // rows + 1

            fig, axs = plt.subplot(rows, cols, figsize=(18, 8))

            for i, img in enumerate(subject_imgs):
                fig.axes[i].imshow(img[:, :, 11], cmap="gray")
                fig.axes[i].axis("off")

            plt.pause(5)
            plt.close()

    @staticmethod
    def check_processed_imgs(file_path: str, phase: str = 'AC'):
        imgs = os.listdir(f"{file_path}/Images/{phase}")
        segs = os.listdir(f"{file_path}/Segmentations/{phase}")
        print(len(imgs), len(segs))

        if len(imgs) >= len(segs):
            for im in imgs:
                if im not in segs:
                    print(f"No segmentation matching {im}")

        else:
            for seg in segs:
                if seg not in imgs:
                    print(f"No image matching {seg}")


class Unpaired(ImgConvBase):

    def __init__(self, **kwargs):  
        super().__init__(**kwargs)
    
    def save_data(
        self,
        subject_ID: list = None,
        HU_min: int = None,
        HU_max: int = None,
        subvol_depth: int = 0,
        down_sample: int = 1,
        file_type: str = "npy"
        ) -> int:

        if subject_ID:
            subjects = subject_ID
        else:
            subjects = self.subjects
        
        count = 1
        total = len(subjects)
    
        if not os.path.exists(f"{self.save_path}/Images"):
            os.makedirs(f"{self.save_path}/Images")

        if not os.path.exists(f"{self.save_path}/Segmentations"):
            os.makedirs(f"{self.save_path}/Segmentations")
        
        for subject in subjects:
            proc = self.load_subject(subject, HU_min=HU_min, HU_max=HU_max)

            if isinstance(proc, tuple):
                imgs = proc[0]
                segs = proc[1]

            elif proc == None:
                print(f"{subject} skipped")
                continue
            
            elif isinstance(proc, dict):
                imgs = proc
                segs = None
            
            else:
                raise ValueError(type(proc))

            if file_type == "npy":
                img_arrays = [itk.GetArrayFromImage(i).transpose([1, 2, 0]).astype("float16") for i in imgs.values()]
                vol_thick = img_arrays[0].shape[2]  

            else:
                img_arrays = imgs.values()
            
            img_names = list(imgs.keys())

            if segs is not None:
                if file_type == "npy":
                    seg_arrays = [itk.GetArrayFromImage(s).transpose([1, 2, 0]).astype("int8") for s in segs.values()]
                
                else:
                    seg_arrays = segs.values()

                seg_names = list(segs.keys())    

            # Partition into sub-volumes
            for name, img in zip(img_names, img_arrays):
                idx = 0  
                stem = name[-16:-5]

                if subvol_depth > 0:
                    assert file_type != "npy", "Only works with npy"

                    for i in range(0, vol_thick, self.output_dims[2]):
                        if i + self.output_dims[2] > vol_thick:
                            break

                        sub_vol = img[::down_sample, ::down_sample, i:i + self.output_dims[2]]
                        np.save(f"{self.save_path}/Images/{stem}_{idx:03d}.npy", sub_vol)
                        idx += 1
                        count += 1

                else:
                    if file_type == "npy":
                        np.save(f"{self.save_path}/Images/{stem}.npy", img[::down_sample, ::down_sample, :])
                    
                    else:
                        assert down_sample == 1, "Down-sample > 1 only with npy"
                        itk.WriteImage(img, f"{self.save_path}/Images/{stem}.nii")

            if segs is not None:

                for name, seg in zip(seg_names, seg_arrays):
                    idx = 0  
                    stem = name[-22:-11]

                    if subvol_depth > 0:
                        assert file_type != "npy", "Only works with npy"

                        for i in range(0, vol_thick, self.output_dims[2]):
                            if i + self.output_dims[2] > vol_thick:
                                break

                            sub_vol = seg[::down_sample, ::down_sample, i:i + self.output_dims[2]]
                            np.save(f"{self.save_path}/Segmentations/{stem}_{idx:03d}.npy", sub_vol)
                            idx += 1
                            count += 1

                    else:
                        if file_type == "npy":
                            np.save(f"{self.save_path}/Segmentations/{stem}.npy", seg[::down_sample, ::down_sample, :])

                        else:
                            assert down_sample == 1, "Down-sample > 1 only with npy"
                            itk.WriteImage(seg, f"{self.save_path}/Segmentations/{stem}.nii")

        return count

    @staticmethod
    def check_processed_imgs(file_path: str):
        imgs = os.listdir(f"{file_path}/Images")
        segs = os.listdir(f"{file_path}/Segmentations")
        print(len(imgs), len(segs))

        for seg in segs:
            if seg not in imgs:
                print(f"No image matching {seg}")

    @staticmethod
    def check_saved(save_path):
        subjects = []

        for img in os.listdir(f"{save_path}/Images"):
            if img[:6] not in subjects: subjects.append(img[:6])

        fig, axs = plt.subplots(4, 4, figsize=(18, 8))

        for img_name in subjects:
            subject_imgs = []
            path = f"{save_path}/Images"
            imgs = os.listdir(path)
            subject_img_names = [im for im in imgs if img_name in im]
            subject_imgs += [np.load(f"{path}/{im}").astype("float32") for im in subject_img_names]
            path = f"{save_path}/Segmentations"
            segs = os.listdir(path)
            subject_seg_names = [seg for seg in segs if img_name in seg]
            subject_imgs += [np.load(f"{path}/{seg}").astype("float32") for seg in subject_seg_names]
            subject_img_names += subject_seg_names

            for i in range(len(subject_imgs)):
                if np.mean(np.square(subject_imgs[i][:, :, 0] - subject_imgs[i][:, :, 1])) > 50000:
                    print(subject_img_names[i])
            for i in range(len(subject_imgs) // 16 + 1):
                name_subset = subject_img_names[(i * 16):((i + 1) * 16)]
                img_subset = subject_imgs[(i * 16):((i + 1) * 16)]

                for j, img in enumerate(img_subset):
                    axs.ravel()[j].imshow(img[:, :, 11], cmap="gray")
                    axs.ravel()[j].set_title(name_subset[j])
                    axs.ravel()[j].axis("off")

                plt.pause(2)
                plt.cla()


if __name__ == "__main__":

    FILE_PATH = "Z:/Clean_CT_Data/Toshiba/"
    SAVE_PATH = "D:/ProjectImages/SyntheticContrastNeedle"

    to_include = ["T051A0", "T052A0", "T055A0", "T057A0", "T058A0", "T061A0", "T062A0", "T063A0", "T064A0", "T065A1", "T066A0", "T067A0", "T068A0", "T069A0", "T070A0"]

    with open("syntheticcontrast_v02/preproc/ignore.json", 'r') as fp:
        ignore = json.load(fp)

    subject_ignore = list(ignore["subject_ignore"].keys())
    image_ignore = ignore["image_ignore"]

    Test = Unpaired(
        image_path=FILE_PATH + "/Images",
        segmentation_path=FILE_PATH + "/Segmentations",
        transformation_path=FILE_PATH + "/Transforms",
        save_path=SAVE_PATH,
        output_dims=(256, 256, 64),
        include=to_include,
        ignore=subject_ignore, NCC_tol=0.0
        )

    # Test.list_images(
    #     ignore=image_ignore,
    #     num_AC=1, num_VC=1, num_HQ=3
    #     ).display(display=True, HU_min=-150, HU_max=250)

    Test.list_images(
        ignore=image_ignore,
        num_AC=1, num_VC=1, num_HQ=None
        ).save_data(HU_min=-500, HU_max=2500, file_type="npy", down_sample=2)

    Unpaired.check_processed_imgs(SAVE_PATH)
    Unpaired.check_saved(SAVE_PATH)
