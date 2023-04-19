import enum
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import SimpleITK as itk  # noqa: N813
import tensorflow as tf
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from vq_sce import ABDO_WINDOW, HU_MAX, HU_MIN
from vq_sce.networks.build_model import build_model
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.build_dataloader import get_test_dataloader
from vq_sce.utils.losses import L1
from vq_sce.utils.patch_utils import CombinePatches, extract_patches, generate_indices

STRIDE_FACTOR = 4


# -------------------------------------------------------------------------


@enum.unique
class Options(str, enum.Enum):
    DISPLAY = "display"
    SAVE = "save"
    METRICS = "metrics"


# -------------------------------------------------------------------------


class Inference(ABC):
    """Base class for performing inference."""

    source_dims: list[int]
    target_dims: list[int]
    strides: list[int]
    patch_size: list[int]

    def __init__(self, config: dict[str, Any], stage: str | None = None):
        self.stage = stage
        self.save_path = config["paths"]["expt_path"] / "predictions"
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.data_path = config["data"]["data_path"]
        self.original_data_path = config["paths"]["original_path"]
        self.mb_size = config["expt"]["mb_size"]
        self.expt_type = config["expt"]["expt_type"]

        self.test_ds, self.TestGenerator = get_test_dataloader(config=config)

        self.model = build_model(config=config, purpose="inference")
        self.combine = CombinePatches()

    @abstractmethod
    def run(self, option: str) -> dict[str, list[float]] | None:
        """Run inference on test data."""
        raise NotImplementedError

    def calc_patches_per_slice(self) -> int:
        """Calculate number of patches for a given slice.
        Assumes square slices of size height * height and no padding.
        """
        height = self.source_dims[1]
        strides = self.strides[1]
        patch_size = self.patch_size[1]

        return ((height - patch_size + strides) // strides) ** 2

    def display(self, pred: npt.NDArray[np.float32], subject_id: str) -> None:
        """Display predicted images."""
        depth, height, _ = pred.shape

        plt.subplot(1, 2, 1)
        plt.imshow(pred[depth // 2, :], cmap="gray", **ABDO_WINDOW)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(np.flipud(pred[:, height // 2, :]), cmap="gray", **ABDO_WINDOW)
        plt.axis("off")
        plt.title(subject_id)
        plt.show()

    def save(
        self,
        pred: npt.NDArray[np.float32],
        source_id: str,
        target_id: str,
    ) -> None:
        """Save predicted images."""
        img_nrrd = itk.GetImageFromArray(pred.astype("int16"))

        if self.original_data_path is not None:
            if (
                self.stage == Task.CONTRAST
                and source_id in self.TestGenerator.source_coords[target_id].keys()
            ):
                original = itk.ReadImage(
                    str(self.original_data_path / source_id[0:6] / f"{source_id}.nrrd"),
                )
                z_offset = self.TestGenerator.source_coords[target_id][source_id][0]

            else:
                base_hq_name = list(self.TestGenerator.source_coords[source_id].keys())[
                    0
                ]
                original = itk.ReadImage(
                    str(
                        self.original_data_path
                        / source_id[0:6]
                        / f"{base_hq_name}.nrrd",
                    ),
                )
                source_coords = list(
                    self.TestGenerator.source_coords[source_id].values(),
                )[0]
                z_offset = source_coords[0]

            img_nrrd.SetDirection(original.GetDirection())
            img_nrrd.SetSpacing(original.GetSpacing())
            origin = original.GetOrigin()
            new_origin = (origin[0], origin[1], origin[2] + z_offset)
            img_nrrd.SetOrigin(new_origin)

        itk.WriteImage(img_nrrd, str(self.save_path / f"{source_id}.nrrd"))
        print(f"{source_id} saved")  # noqa: T201

    def calc_metrics(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> dict[str, float]:
        """Calculate MSE, pSNR, SSIM between predicted and ground truth image."""
        metrics = {
            "MSE": mean_squared_error(target, pred),
            "pSNR": peak_signal_noise_ratio(target, pred, data_range=HU_MAX - HU_MIN),
            "SSIM": structural_similarity(target, pred, data_range=HU_MAX - HU_MIN),
        }

        return metrics


# -------------------------------------------------------------------------


class SingleScaleInference(Inference):
    """Perform inference on images with single scale/patch-based models."""

    def __init__(self, config: dict[str, Any], stage: str | None = None):
        super().__init__(config, stage)

        self.stage = stage

        if stage == Task.CONTRAST:
            depth, height, width = config["data"]["target_dims"]
        else:
            depth, height, width = config["data"]["source_dims"]

        self.source_dims = config["data"]["source_dims"]
        self.target_dims = config["data"]["target_dims"]
        self.scale = config["hyperparameters"]["scales"][0]
        self.patch_size = [depth, height // self.scale, width // self.scale]

        if stage == Task.CONTRAST and self.scale == 1:
            self.strides = [depth // STRIDE_FACTOR, 1, 1]
        elif stage == Task.CONTRAST and self.scale > 1:
            self.strides = [
                depth // STRIDE_FACTOR,
                self.patch_size[1] // STRIDE_FACTOR,
                self.patch_size[2] // STRIDE_FACTOR,
            ]
        elif stage == Task.SUPER_RES and self.scale == 1:
            self.strides = [1, 1, 1]
        elif stage == Task.SUPER_RES and self.scale > 1:
            self.strides = [
                1,
                self.patch_size[1] // STRIDE_FACTOR,
                self.patch_size[2] // STRIDE_FACTOR,
            ]

        self.patches_per_slice = self.calc_patches_per_slice()

    def run(self, option: str) -> dict[str, list[float]] | None:
        """Run inference on test data."""
        metrics: dict[str, list[float]] = {
            "id": [],
            "L1": [],
            "MSE": [],
            "pSNR": [],
            "SSIM": [],
        }

        for data in self.test_ds:
            source = data["source"][0, ...]
            subject_id = data["source_id"][0].numpy().decode("utf-8")
            target_id = data["target_id"][0].numpy().decode("utf-8")

            if self.stage == Task.CONTRAST:
                self.combine.new_subject(source.shape)
            elif self.stage == Task.SUPER_RES:
                self.combine.new_subject(self.target_dims)

            # Generate indices of individual patches to sample
            linear_indices = generate_indices(
                source.shape,
                self.strides,
                self.patch_size,
            )
            pred_stack = []

            # To avoid OOM errors, process patches in batches
            num_patches = len(linear_indices) // self.patches_per_slice

            for i in range(0, len(linear_indices), num_patches):
                batch_indices = linear_indices[i : i + num_patches]
                patch_stack = extract_patches(source, batch_indices, self.patch_size)
                stack_depth = patch_stack.shape[0]

                for j in range(0, stack_depth, self.mb_size):
                    if self.expt_type == Task.JOINT:
                        pred_mb, _ = self.model(
                            patch_stack[j : j + self.mb_size, ...],
                            self.stage,
                        )
                    else:
                        pred_mb, _ = self.model(patch_stack[j : j + self.mb_size, ...])

                    pred_mb = self.TestGenerator.un_normalise(pred_mb)

                    # TF stores pred_stack elements on GPU (leading to OOM)
                    with tf.device("CPU:0"):
                        pred_stack.extend(pred_mb[:, :, :, :, 0])

            # Super-resolution requires updated indices for super-resolved image
            if self.stage == Task.SUPER_RES:
                z_scaling = self.target_dims[0] // self.source_dims[0]
                new_strides = [self.strides[0] * z_scaling] + self.strides[1:]
                new_patch_size = [self.patch_size[0] * z_scaling] + self.patch_size[1:]
                linear_indices = generate_indices(
                    self.target_dims,
                    new_strides,
                    new_patch_size,
                )

            # To avoid OOM errors, recombine patches in host memory
            with tf.device("CPU:0"):
                for i in range(0, len(linear_indices), self.patches_per_slice):
                    batch_indices = linear_indices[i : i + self.patches_per_slice]
                    pred_batch = tf.stack(
                        pred_stack[i : i + self.patches_per_slice],
                        axis=0,
                    )
                    self.combine.apply_patches(pred_batch, batch_indices)

            pred = self.combine.get_img().numpy()

            if option == Options.SAVE:
                self.save(pred, subject_id, target_id)

            elif option == Options.DISPLAY:
                self.display(pred, subject_id)

            elif option == Options.METRICS:
                target = self.TestGenerator.un_normalise(data["target"][0, ...])

                metrics["id"].append(subject_id)
                metrics["L1"].append(float(L1(target.astype("float32"), pred)))

                detailed_metrics = self.calc_metrics(pred, target)
                for k, v in detailed_metrics.items():
                    metrics[k].append(v)

                print(f"{subject_id} metrics processed")  # noqa: T201

            else:
                raise ValueError(f"Option not recognised: {option}")

        if option == Options.METRICS:
            return metrics
        else:
            return None


# -------------------------------------------------------------------------


class MultiScaleInference(Inference):
    """Perform inference on images with multi-scale models."""

    def __init__(self, config: dict[str, Any], stage: str | None = None):
        super().__init__(config, stage)

        depth, height, width = config["data"]["target_dims"]
        scales = config["hyperparameters"]["scales"]
        self.patch_size = [depth, height // scales[0], width // scales[0]]
        self.upscale_patch_size = [
            self.patch_size[0],
            self.patch_size[1] * 2,
            self.patch_size[2] * 2,
        ]
        self.strides = [
            1,
            self.patch_size[1] // STRIDE_FACTOR,
            self.patch_size[2] // STRIDE_FACTOR,
        ]
        self.upscale_strides = [
            self.strides[0],
            self.strides[1] * 2,
            self.strides[2] * 2,
        ]
        self.dn_samp = config["hyperparameters"]["scales"][0]
        self.num_scales = len(config["hyperparameters"]["scales"])

    def run(self, option: str) -> dict[str, list[float]] | None:
        """Run inference on test data."""
        metrics: dict[str, list[float]] = {
            "id": [],
            "L1": [],
            "MSE": [],
            "pSNR": [],
            "SSIM": [],
        }

        for data in self.test_ds:
            source = data["source"][0, ...]
            source = source[:, :: self.dn_samp, :: self.dn_samp]
            subject_id = data["subject_id"][0].numpy().decode("utf-8")
            target_id = data["target_id"][0].numpy().decode("utf-8")

            linear_coords = generate_indices(
                source.shape,
                self.strides,
                self.patch_size,
            )
            patch_stack = extract_patches(source, linear_coords, self.patch_size)
            num_patches = patch_stack.shape[0]
            pred_stack = []

            for i in range(0, num_patches, self.mb_size):
                pred_mb, _ = self.model(patch_stack[i : (i + self.mb_size), ...])
                pred_stack.extend(pred_mb[:, :, :, :, 0])

            pred_stack = tf.stack(pred_stack, axis=0)
            depth, height, width = source.shape[1:-1]
            upscale_dims = (depth, height * 2, width * 2)
            self.combine.new_subject(upscale_dims)

            upscale_linear_coords = generate_indices(
                upscale_dims,
                self.upscale_strides,
                self.upscale_patch_size,
            )
            self.combine.apply_patches(pred_stack, upscale_linear_coords)
            pred = self.combine.get_img()

            for _ in range(1, self.num_scales - 1):
                upscale_dims = (
                    upscale_dims[0],
                    upscale_dims[1] * 2,
                    upscale_dims[2] * 2,
                )
                self.combine.new_subject(upscale_dims)

                linear_coords = generate_indices(
                    pred.shape,
                    self.strides,
                    self.patch_size,
                )
                patch_stack = extract_patches(pred, linear_coords, self.patch_size)
                num_patches = patch_stack.shape[0]
                pred_stack = []

                for i in range(0, num_patches, self.mb_size):
                    pred_mb, _ = self.model(patch_stack[i : (i + self.mb_size), ...])
                    pred_stack.extend(pred_mb)

                pred_stack = tf.stack(pred_stack, axis=0)
                upscale_linear_coords = generate_indices(
                    upscale_dims,
                    self.upscale_strides,
                    self.upscale_patch_size,
                )
                self.combine.apply_patches(pred_stack, upscale_linear_coords)
                pred = self.combine.get_img()

            pred = self.TestGenerator.un_normalise(pred.numpy())

            if option == Options.SAVE:
                self.save(pred, subject_id, target_id)

            elif option == Options.DISPLAY:
                self.display(pred, subject_id)

            elif option == Options.METRICS:
                target = self.TestGenerator.un_normalise(data["target"][0, ...])

                metrics["id"].append(subject_id)
                metrics["L1"].append(float(L1(target.astype("float32"), pred)))

                detailed_metrics = self.calc_metrics(pred, target)
                for k, v in detailed_metrics.items():
                    metrics[k].append(v)

                print(f"{subject_id} metrics processed")  # noqa: T201

            else:
                raise ValueError(f"Option not recognised: {option}")

        if option == Options.METRICS:
            return metrics
        else:
            return None
