import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import os
import SimpleITK as itk


#-------------------------------------------------------------------------

def process_time(time: str):
    ms = float(time.split('.')[1]) / 1000.0
    hms = time.split('.')[0]
    assert len(hms) == 6
    h = int(hms[0:2])
    m = int(hms[2:4])
    s = int(hms[4:6])

    return 3600 * h + 60 * m + s + ms


#-------------------------------------------------------------------------

def seg_mean(img, seg):
    return (img * seg).sum() / (seg.sum() + 1e-8)


#-------------------------------------------------------------------------

def load_images(subject_name, img_path, seg_path, trans_path, ignore):
    images = {}
    segs = {}

    ACs = [f"{img_path}/{subject_name}/{i}" for i in os.listdir(f"{img_path}/{subject_name}") if 'AC' in i]
    VCs = [f"{img_path}/{subject_name}/{i}" for i in os.listdir(f"{img_path}/{subject_name}") if 'VC' in i]
    HQs = [f"{img_path}/{subject_name}/{i}" for i in os.listdir(f"{img_path}/{subject_name}") if 'HQ' in i]

    if len(ACs) != 1 or len(VCs) != 1:
        print(subject_name, len(ACs), len(VCs))

    image_names = ACs + VCs + HQs
    image_names = [n for n in image_names if n[-16:-5] not in ignore]

    for i in range(len(image_names)):
        transform_candidates = glob.glob(f"{trans_path}/{subject_name}/{image_names[i][-8:-5]}_to_*")

        if len(transform_candidates) == 1:
            transform = itk.ReadTransform(transform_candidates[0])
        elif len(transform_candidates) == 0:
            transform = None
        else:
            raise ValueError(transform_candidates)

        img = itk.ReadImage(image_names[i], itk.sitkInt32)

        if transform is not None:
            img = itk.Resample(img, transform, defaultPixelValue=-2048)

        image_dir = np.around(img.GetDirection())
        assert img.GetSpacing()[2] == 1.0, f"{image_names[i]}: {img.GetSpacing()}"
        seg_name = f"{seg_path}/{subject_name}/{image_names[i][-16:-5]}-label.nrrd"

        try:
            seg = itk.ReadImage(seg_name)

            if transform is not None:
                seg = itk.Resample(seg, transform, defaultPixelValue=0)

        except RuntimeError:
            seg = None
        
        else:
            assert np.isclose(img.GetSpacing(), seg.GetSpacing()).all() and np.isclose(img.GetDirection(), seg.GetDirection()).all(), f"{image_names[i]}: {img.GetSpacing()}, {seg.GetSpacing()}, {img.GetDirection()}, {seg.GetDirection()}"

        # Check image is orientated correctly and flip/rotate if necessary
        if image_dir[0] == 0.0 or image_dir[4] == 0.0:
            img = itk.PermuteAxes(img, [1, 0, 2])
            image_dir = np.around(img.GetDirection())
            img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

            if seg is not None:
                seg = itk.PermuteAxes(seg, [1, 0, 2])
                seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                segs[seg_name[-22:-11]] = seg

        else:
            img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

            if seg is not None:
                seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                segs[seg_name[-22:-11]] = seg

        images[image_names[i][-16:-5]] = img

    return images, segs


#-------------------------------------------------------------------------

def resample(images, segs):
    image_bounds = []

    # Get start/end coords data for images
    for key in images.keys():
        image_dim_z = images[key].GetSize()[2]
        image_origin_z = images[key].GetOrigin()[2]

        # TODO: Will this work without rounding?
        image_bounds.append(np.around([image_origin_z, image_origin_z + image_dim_z - 1]).astype(np.int32))

    image_bounds = np.vstack(image_bounds)
    tightest_bounds = [image_bounds[:, 0].max(), image_bounds[:, 1].min()]

    for i, key in enumerate(images.keys()):
        start = tightest_bounds[0] - image_bounds[i, 0]
        end = tightest_bounds[1] - image_bounds[i, 1]

        if end == 0:
            images[key] = images[key][:, :, start:]
        else:
            images[key] = images[key][:, :, start:end]

        if images[key].GetSize()[2] == 0:
            print(f"{key} img: size {images[key].GetSize()}")

    for i, key in enumerate(segs.keys()):
        # TODO: Will this work without rounding?
        seg_bounds = np.around([segs[key].GetOrigin()[2], segs[key].GetOrigin()[2] + segs[key].GetSize()[2] - 1]).astype(np.int32)
        start = tightest_bounds[0] - seg_bounds[0]
        end = tightest_bounds[1] - seg_bounds[1]

        if end == 0:
            segs[key] = segs[key][:, :, start:]
        else:
            segs[key] = segs[key][:, :, start:end]

        if segs[key].GetSize()[2] == 0:
            print(f"{key} seg: size {segs[key].GetSize()}")

    # Resample source to target coords
    for i in range(1, len(images)):
        key = list(images.keys())[i]
        images[key] = itk.GetArrayFromImage(itk.Resample(images[key], images[list(images.keys())[0]], defaultPixelValue=-2048)).transpose([1, 2, 0])
        assert images[key].shape == images[list(images.keys())[0]].GetSize()

    for i in range(len(segs)):
        key = list(segs.keys())[i]
        segs[key] = itk.GetArrayFromImage(itk.Resample(segs[key], images[list(images.keys())[0]], defaultPixelValue=0)).transpose([1, 2, 0])
        assert segs[key].shape == images[list(images.keys())[0]].GetSize()

    images[list(images.keys())[0]] = itk.GetArrayFromImage(images[list(images.keys())[0]]).transpose([1, 2, 0])

    return images, segs


#-------------------------------------------------------------------------

def display_imgs(imgs, segs, keys, overlay=None, depth_idx=None):
    if overlay is not None:
        overlay[overlay > 0] = 1
        mask = np.ma.masked_where(overlay == False, overlay)

    if depth_idx is None:
        mid_point = imgs[keys[0]].shape[2] // 2
    else:
        mid_point = depth_idx

    if len(keys) % 2 == 0:
        fig, axs = plt.subplots(4, len(keys) // 2, figsize=(18, 10))
        offset = len(keys) // 2
    else:
        fig, axs = plt.subplots(4, len(keys) // 2 + 1, figsize=(18, 10))
        offset = len(keys) // 2 + 1
        axs[0, len(keys) // 2].axis("off")
        axs[1, len(keys) // 2].axis("off")

    for i in range(len(keys) // 2):
        img = imgs[keys[i]]
        axs[0, i].imshow(img[:, :, mid_point], cmap="gray", vmin=-150, vmax=250)
        axs[0, i].set_title(keys[i])

        if overlay is not None:
            axs[0, i].imshow(mask[:, :, mid_point], alpha=0.3, cmap='Set1')
                

        axs[0, i].axis("off")

        try:
            seg = segs[keys[i]]
        except KeyError:
            pass
        else:
            axs[1, i].imshow(seg[:, :, mid_point], cmap="gray")
        finally:
            axs[1, i].axis("off")

    for i in range(offset):
        img = imgs[keys[i + len(keys) // 2]]
        axs[2, i].imshow(img[:, :, mid_point], cmap="gray", vmin=-150, vmax=250)
        axs[2, i].set_title(keys[i + len(keys) // 2])

        if overlay is not None:
            axs[2, i].imshow(mask[:, :, mid_point], alpha=0.3, cmap='Set1')

        axs[2, i].axis("off")

        try:
            seg = segs[keys[i + len(keys) // 2]]
        except KeyError:
            pass
        else:
            axs[3, i].imshow(seg[:, :, mid_point], cmap="gray")
        finally:
            axs[3, i].axis("off")    

    plt.show()


#-------------------------------------------------------------------------

def get_HUs(imgs: dict, seg: object, keys: list):
    Ao = []
    RK = []
    LK = []
    Tu = []

    assert seg.min() == 0 and seg.max() == 4, (seg.min(), seg.max())

    for key in keys:
        Ao.append(seg_mean(imgs[key], seg * (seg == 1)))
        RK.append(seg_mean(imgs[key], seg * (seg == 2)))
        LK.append(seg_mean(imgs[key], seg * (seg == 3)))
        Tu.append(seg_mean(imgs[key], seg * (seg == 4)))

    return Ao, RK, LK, Tu


#-------------------------------------------------------------------------

def aggregate_HUs(
    subject_list: list,
    subject_ignore: list = [],
    image_ignore: list = [],
    times: dict = None,
    img_path: str = None,
    seg_path: str = None,
    trans_path: str = None,
    time_path: str = None
    ):
    HUs = {}
    fewest_series = 1000

    for subject in subject_list:
        if subject in subject_ignore or 'F' in subject:
            continue

        print(subject)
        imgs, segs = load_images(subject, img_path, seg_path, trans_path, ignore=image_ignore)
        imgs, segs = resample(imgs, segs)

        AC = [n for n in imgs.keys() if 'AC' in n]
        VC = [n for n in imgs.keys() if 'VC' in n]
        HQ = [n for n in imgs.keys() if 'HQ' in n]
        keys = sorted(AC + VC + HQ, key=lambda x: int(x[-3:]))
        if 'AC' not in keys[1]: continue

        with open(f"{time_path}/{subject}/time.json", 'r') as fp:
            times = json.load(fp)

        # if len(keys) < fewest_series:
        #     fewest_series = len(keys)

        Ao, RK, LK, Tu = get_HUs(imgs, segs[AC[0]], keys)

        t = [times[f"{k}.nrrd"] for k in keys]
        assert len(t) == len(Ao), f"Times length != contrast length {len(t)} {len(Ao)}"

        # if times is not None:
        #     t = [times[f"{k}.nrrd"] for k in keys]
        #     assert len(t) == len(Ao), f"Times length != contrast length {len(t)} {len(Ao)}"
        # else:
        #     t = None

        HUs[subject] = {'Ao': Ao, 'RK': RK, 'LK': LK, 'Tu': Tu, "times": t}
    
    # HU_agg = {
    #     'Ao': np.vstack([HUs[key]['Ao'][0:fewest_series] for key in HUs.keys()]),
    #     'RK': np.vstack([HUs[key]['RK'][0:fewest_series] for key in HUs.keys()]),
    #     'LK': np.vstack([HUs[key]['LK'][0:fewest_series] for key in HUs.keys()]),
    #     'Tu': np.vstack([HUs[key]['Tu'][0:fewest_series] for key in HUs.keys()]),
    #     'times': np.vstack([HUs[key]['times'][0:fewest_series] for key in HUs.keys()])
    #     }

    return HUs
