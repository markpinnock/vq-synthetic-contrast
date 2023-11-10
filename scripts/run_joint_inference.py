import json
from pathlib import Path

import numpy as np
import SimpleITK as itk  # noqa: N813

from vq_sce import HU_MAX, HU_MIN
from vq_sce.networks.build_model import build_model_inference


def normalise(img):
    norm_img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    norm_img = 2 * norm_img - 1
    return norm_img


def un_normalise(img):
    img = (img + 1) / 2
    img = img * (HU_MAX - HU_MIN) + HU_MIN
    return img


def save(pred, source_coords, source_id, original_data_path, save_path):
    img_nrrd = itk.GetImageFromArray(pred.astype("int16"))

    base_hq_name = list(source_coords[source_id].keys())[0]
    original = itk.ReadImage(
        str(original_data_path / source_id[0:6] / f"{base_hq_name}.nrrd"),
    )
    z_offset = list(source_coords[source_id].values())[0][0]

    img_nrrd.SetDirection(original.GetDirection())
    img_nrrd.SetSpacing(original.GetSpacing())
    origin = original.GetOrigin()
    new_origin = (origin[0], origin[1], origin[2] + z_offset)
    img_nrrd.SetOrigin(new_origin)

    itk.WriteImage(img_nrrd, str(save_path / f"{source_id}.nrrd"))


expt_path = Path(
    "/path/to/expts",
)
data_path = Path("/path/to/data")
original_data_path = Path("/path/to/original/images")

imgs = [
    "T005A0LQ055.npy",
    "T065A1LQ031.npy",
    "T066A0LQ111.npy",
    "T069A0LQ065.npy",
    "T086A0LQ048.npy",
    "T088A0LQ150.npy",
    "T105A0LQ096.npy",
    "T107A0LQ067.npy",
    "T115A0LQ151.npy",
    "T126A0LQ050.npy",
]

with open(data_path / "source_coords.json") as fp:
    source_coords = json.load(fp)

sr_expt = "sr_vq0"
ce_expt = "ce_vq0"
sr_kwargs = {"expt_path": expt_path / sr_expt, "epoch": 200}
ce_kwargs = {"expt_path": expt_path / ce_expt, "epoch": 2000}

sr_model = build_model_inference(**sr_kwargs)
ce_model = build_model_inference(**ce_kwargs)
save_path = expt_path / ce_expt / "predictions-joint"
save_path.mkdir(exist_ok=True)

for img in imgs:
    lq_img = np.load(data_path / "LQ" / img)[np.newaxis, :, :, :, np.newaxis]
    lq_img = normalise(lq_img)
    hq_pred = sr_model(lq_img)
    ce_pred = ce_model(hq_pred).numpy()[0, :, :, :, 0]
    ce_pred = un_normalise(ce_pred)

    save(ce_pred, source_coords, img[:-4], original_data_path, save_path)

sr_expt = "sr_vq128-128-256-512-512"
ce_expt = "ce_vq128-128-256-512-512"
sr_kwargs = {"expt_path": expt_path / sr_expt, "epoch": 400}
ce_kwargs = {"expt_path": expt_path / ce_expt, "epoch": 4000}

sr_model = build_model_inference(**sr_kwargs)
ce_model = build_model_inference(**ce_kwargs)
save_path = expt_path / ce_expt / "predictions-joint"
save_path.mkdir(exist_ok=True)

for img in imgs:
    lq_img = np.load(data_path / "LQ" / img)[np.newaxis, :, :, :, np.newaxis]
    lq_img = normalise(lq_img)
    hq_pred = sr_model(lq_img)
    ce_pred = ce_model(hq_pred).numpy()[0, :, :, :, 0]
    ce_pred = un_normalise(ce_pred)

    save(ce_pred, source_coords, img[:-4], original_data_path, save_path)

jo_expt = "jo_vq128-128-512-512-1024"
jo_kwargs = {"expt_path": expt_path / jo_expt, "epoch": 4000}
jo_model = build_model_inference(**jo_kwargs)
save_path = expt_path / jo_expt / "predictions-joint"
save_path.mkdir(exist_ok=True)

for img in imgs:
    lq_img = np.load(data_path / "LQ" / img)[np.newaxis, :, :, :, np.newaxis]
    lq_img = normalise(lq_img)
    ce_pred = jo_model(lq_img).numpy()[0, :, :, :, 0]
    ce_pred = un_normalise(ce_pred)

    save(ce_pred, source_coords, img[:-4], original_data_path, save_path)

jo_expt = "jo_vq128-64-256-32-512"
jo_kwargs = {"expt_path": expt_path / jo_expt, "epoch": 4000}
jo_model = build_model_inference(**jo_kwargs)
save_path = expt_path / jo_expt / "predictions-joint"
save_path.mkdir(exist_ok=True)

for img in imgs:
    lq_img = np.load(data_path / "LQ" / img)[np.newaxis, :, :, :, np.newaxis]
    lq_img = normalise(lq_img)
    ce_pred = jo_model(lq_img).numpy()[0, :, :, :, 0]
    ce_pred = un_normalise(ce_pred)

    save(ce_pred, source_coords, img[:-4], original_data_path, save_path)
