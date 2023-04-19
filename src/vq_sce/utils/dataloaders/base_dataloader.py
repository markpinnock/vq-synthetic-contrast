from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, TypedDict

import numpy as np
import numpy.typing as npt

from vq_sce import HU_MAX, HU_MIN

# -------------------------------------------------------------------------


class DataDictType(TypedDict):
    source: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]


# -------------------------------------------------------------------------


class BaseDataloader(ABC):
    _config: dict[str, Any]
    _dataset_type: str
    _down_sample: int
    _sources: dict[str, Path]
    _targets: dict[str, Path]
    _ex_sources: npt.NDArray[np.float32]
    _ex_targets: npt.NDArray[np.float32]
    _source_coords: dict[str, dict[str, list[int]]]

    @abstractmethod
    def __init__(self) -> None:
        pass

    def _train_val_split(self) -> None:
        """Get unique subject IDs for subject-level train/val split."""
        # Need procedure IDs (as poss. >1 per subject)
        unique_ids = []
        for img_id in self._targets.keys():
            if img_id[0:4] not in unique_ids:
                unique_ids.append(img_id[0:4])

        assert (
            self._config["fold"] < self._config["cv_folds"]
        ), f"Fold number {self._config['fold']}of {self._config['cv_folds']} folds"

        N = len(unique_ids)  # noqa: N806
        np.random.shuffle(unique_ids)

        # Split into folds by subject
        assert (
            self._config["cv_folds"] > 0
        ), f"Number of folds: {self._config['cv_folds']}"
        if self._config["cv_folds"] > 1:
            num_in_fold = N // self._config["cv_folds"]
            partition_1 = self._config["fold"] * num_in_fold
            partition_2 = (self._config["fold"] + 1) * num_in_fold

            if self._dataset_type == "training":
                fold_ids = unique_ids[0:partition_1] + unique_ids[partition_2:]
            elif self._dataset_type == "validation":
                fold_ids = unique_ids[partition_1:partition_2]
            else:
                raise ValueError("Select 'training' or 'validation'")

            self._sources = {
                k: v for k, v in self._sources.items() if k[0:4] in fold_ids
            }
            self._targets = {
                k: v for k, v in self._targets.items() if k[0:4] in fold_ids
            }

    @abstractmethod
    def _generate_example_images(self) -> None:
        pass

    def _preprocess_image(
        self,
        img: npt.NDArray[np.float16],
        lower: int | None,
        upper: int | None,
    ) -> npt.NDArray[np.float32]:
        img = img[lower:upper, :: self._down_sample, :: self._down_sample]
        proc_img = self._normalise(img)

        return proc_img

    def _normalise(self, img: npt.NDArray[np.float16]) -> npt.NDArray[np.float32]:
        norm_img = (img - HU_MIN) / (HU_MAX - HU_MIN)
        norm_img = 2 * norm_img - 1
        return norm_img

    def un_normalise(self, img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        img = (img + 1) / 2
        img = img * (HU_MAX - HU_MIN) + HU_MIN
        return img

    @abstractmethod
    def data_generator(self) -> Iterator[DataDictType]:
        pass

    @abstractmethod
    def inference_generator(self) -> Iterator[dict[str, Any]]:
        pass

    @property
    def example_images(self) -> DataDictType:
        """Return example images."""
        return {"target": self._ex_targets, "source": self._ex_sources}

    @property
    def data(self) -> dict[str, dict[str, Path]]:
        """Return list of all images."""
        return {"target": self._targets, "source": self._sources}

    @property
    def source_coords(self) -> dict[str, dict[str, list[int]]]:
        """Return source coords."""
        return self._source_coords
