from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Any

import matplotlib.pyplot as plt
import SimpleITK as itk

from vq_sce import ABDO_WINDOW
from vq_sce.networks.build_model import build_model
from vq_sce.networks.model import Task
from vq_sce.utils.dataloaders.build_dataloader import get_test_dataloader
from vq_sce.utils.patch_utils import generate_indices, extract_patches, CombinePatches

STRIDE_FACTOR = 4


#-------------------------------------------------------------------------


class Inference(ABC):
    """Base class for performing inference."""

    source_dims: list[int]
    target_dims: list[int]
    strides: list[int]
    patch_size: list[int]

    def __init__(self, config: dict[str, Any]) -> None:
        self.save_path = config["paths"]["expt_path"] / "predictions"
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.mb_size = config["expt"]["mb_size"]
        self.task = config["data"]["type"]

        self.test_ds, self.TestGenerator = get_test_dataloader(config=config)

        self.model = build_model(config=config, purpose="inference")
        self.combine = CombinePatches()

    @abstractmethod
    def run(self, save: bool) -> None:
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

    def save(self, pred: npt.NDArray[np.float32], subject_id: str) -> None:
        """Save predicted images."""
        img_nrrd = itk.GetImageFromArray(pred.astype("int16").transpose([2, 0, 1]))
        itk.WriteImage(img_nrrd, str(self.save_path / f"{subject_id}.nrrd"))
        print(f"{subject_id} saved")


#-------------------------------------------------------------------------


class SingleScaleInference(Inference):
    """Perform inference on images with single scale/patch-based models."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        depth, height, width = config["data"]["source_dims"]
        self.source_dims = config["data"]["source_dims"]
        self.target_dims = config["data"]["target_dims"]
        self.scale = config["hyperparameters"]["scales"][0]
        self.patch_size = [depth, height // self.scale, width // self.scale]

        if self.task == Task.CONTRAST and self.scale == 1:
            self.strides = [depth // STRIDE_FACTOR, 1, 1]
        elif self.task == Task.CONTRAST and self.scale > 1:
            self.strides = [depth // STRIDE_FACTOR, self.patch_size[1] // STRIDE_FACTOR, self.patch_size[2] // STRIDE_FACTOR]
        elif self.task == Task.SUPER_RES and self.scale == 1:
            self.strides = [1, 1, 1]
        elif self.task == Task.SUPER_RES and self.scale > 1:
            self.strides = [1, self.patch_size[1] // STRIDE_FACTOR, self.patch_size[2] // STRIDE_FACTOR]
        elif self.task == Task.JOINT and self.scale == 1:
            self.strides = [1, 1, 1]
        elif self.task == Task.JOINT and self.scale > 1:
            self.strides = [1, self.patch_size[1] // STRIDE_FACTOR, self.patch_size[2] // STRIDE_FACTOR]

        self.patches_per_slice = self.calc_patches_per_slice()

    def run(self, save: bool) -> None:
        """Run inference on test data."""

        for data in self.test_ds:
            source = data["source"][0, ...]
            subject_id = data["subject_id"][0].numpy().decode("utf-8")

            if self.task == Task.CONTRAST:
                self.combine.new_subject(source.shape)
            elif self.task == Task.SUPER_RES:
                self.combine.new_subject(self.target_dims)

            # Generate indices of individual patches to sample
            linear_indices = generate_indices(source.shape, self.strides, self.patch_size)
            pred_stack = []

            # To avoid OOM errors, process patches in batches
            num_patches = len(linear_indices) // self.patches_per_slice

            for i in range(0, len(linear_indices), num_patches):
                batch_indices = linear_indices[i:i + num_patches]
                patch_stack = extract_patches(source, batch_indices, self.patch_size)
                stack_depth = patch_stack.shape[0]

                for j in range(0, stack_depth, self.mb_size):
                    pred_mb, _ = self.model(patch_stack[j:j + self.mb_size, ...])
                    pred_mb = self.TestGenerator.un_normalise(pred_mb)
                    pred_stack.extend(pred_mb[:, :, :, :, 0])

            # Super-resolution requires updated indices for super-resolved image
            if self.task == Task.SUPER_RES:
                z_scaling = self.target_dims[0] // self.source_dims[0]
                new_strides = [self.strides[0] * z_scaling] + self.strides[1:]
                new_patch_size = [self.patch_size[0] * z_scaling] + self.patch_size[1:]
                linear_indices = generate_indices(self.target_dims, new_strides, new_patch_size)

            # To avoid OOM errors, recombine patches in batches
            for i in range(0, len(linear_indices), self.patches_per_slice):
                batch_indices = linear_indices[i:i + self.patches_per_slice]
                pred_batch = tf.stack(pred_stack[i:i + self.patches_per_slice], axis=0)
                self.combine.apply_patches(pred_batch, batch_indices)
    
            pred = self.combine.get_img().numpy()

            if save:
                self.save(pred, subject_id)
            else:
                self.display(pred, subject_id)


#-------------------------------------------------------------------------


class MultiScaleInference(Inference):
    """Perform inference on images with multi-scale models."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        depth, height, width = config["data"]["target_dims"]
        scales = config["hyperparameters"]["scales"]
        self.patch_size = [depth, height // scales[0], width // scales[0]]
        self.upscale_patch_size = [self.patch_size[0], self.patch_size[1] * 2, self.patch_size[2] * 2]
        self.upscale_strides = [STRIDES[0], STRIDES[1] * 2, STRIDES[2] * 2]
        self.dn_samp = config["hyperparameters"]["scales"][0]
        self.num_scales = len(config["hyperparameters"]["scales"])

    def run(self, save: bool) -> None:
        """Run inference on test data."""
        for data in self.test_ds:
            source = data["source"][0, ...]
            source = source[:, ::self.dn_samp, ::self.dn_samp]
            # source = tf.transpose(source, [2, 0, 1])                    # TODO CHANGE
            subject_id = data["subject_id"][0].numpy().decode("utf-8")

            linear_coords = generate_indices(source.shape, STRIDES, self.patch_size)
            patch_stack = extract_patches(source, linear_coords, self.patch_size)
            # patch_stack = tf.transpose(patch_stack, [0, 2, 3, 1, 4])    # TODO CHANGE
            num_patches = patch_stack.shape[0]
            pred_stack = []

            for i in range(0, num_patches, self.mb_size):
                pred_mb, _ = self.model(patch_stack[i:(i + self.mb_size), ...])
                pred_stack.extend(pred_mb[:, :, :, :, 0])

            pred_stack = tf.stack(pred_stack, axis=0)
            # pred_stack = tf.transpose(pred_stack, [0, 3, 1, 2])    # TODO CHANGE
            depth, height, width = source.shape[1:-1]
            upscale_dims = (depth, height * 2, width * 2)
            self.combine.new_subject(upscale_dims)

            upscale_linear_coords = generate_indices(upscale_dims, self.upscale_strides, upscale_patch_size)
            self.combine.apply_patches(pred_stack, upscale_linear_coords)
            pred = self.combine.get_img()
            # pred = tf.transpose(pred, [1, 2, 0]).numpy()    # TODO CHANGE

            for _ in range(1, self.num_scales - 1):
                # pred = tf.transpose(pred, [2, 0, 1])                    # TODO CHANGE
                upscale_dims = (upscale_dims[0], upscale_dims[1] * 2, upscale_dims[2] * 2)
                self.combine.new_subject(upscale_dims)

                linear_coords = generate_indices(pred.shape, self.strides, self.patch_size)
                patch_stack = extract_patches(pred, linear_coords, self.patch_size)
                # patch_stack = tf.transpose(patch_stack, [0, 2, 3, 1, 4])    # TODO CHANGE
                num_patches = patch_stack.shape[0]
                pred_stack = []

                for i in range(0, num_patches, self.mb_size):
                    pred_mb, _ = self.model(patch_stack[i:(i + self.mb_size), ...])
                    pred_stack.extend(pred_mb)

                pred_stack = tf.stack(pred_stack, axis=0)
                # pred_stack = tf.transpose(pred_stack, [0, 3, 1, 2])    # TODO CHANGE
                upscale_linear_coords = generate_indices(upscale_dims, self.upscale_strides, self.upscale_patch_size)
                self.combine.apply_patches(pred_stack, upscale_linear_coords)
                pred = self.combine.get_img()
                # pred = tf.transpose(pred, [1, 2, 0])    # TODO CHANGE

            pred = self.TestGenerator.un_normalise(pred.numpy())

            if save:
                self.save(pred, subject_id)
            else:
                self.display(pred, subject_id)
