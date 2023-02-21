from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from vq_sce import HU_MIN, HU_MAX, LQ_DEPTH, LQ_SLICE_THICK


class BaseDataloader(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def _train_val_split(self) -> None:
        """ Get unique subject IDs for subject-level train/val split """

        # Need procedure IDs (as poss. >1 per subject)
        unique_ids = []
        for img_id in self._targets.keys():
            if img_id[0:4] not in unique_ids:
                unique_ids.append(img_id[0:4])

        assert self._config["fold"] < self._config["cv_folds"], (
                f"Fold number {self._config['fold']}"
                f"of {self._config['cv_folds']} folds"
            )

        N = len(unique_ids)
        np.random.shuffle(unique_ids)

        # Split into folds by subject
        assert self._config["cv_folds"] > 0, f"Number of folds: {self._config['cv_folds']}"
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

            self._sources = {k: v for k, v in self._sources.items() if k[0:4] in fold_ids}
            self._targets = {k: v for k, v in self._targets.items() if k[0:4] in fold_ids}

    @abstractmethod
    def _generate_example_images(self):
        pass

    def _calc_coords(
        self,
        source_id: str,
        target_id: str
    ) -> tuple[list[int], list[int]]:

        source_coords = list(self._source_coords[source_id].values())[0]
        target_coords = list(self._source_coords[target_id].values())[0]
        target_coords = [
            source_coords[0] - target_coords[0],
            source_coords[0] - target_coords[0] + (LQ_SLICE_THICK * LQ_DEPTH)
        ]
    
        return source_coords, target_coords

    def _preprocess_image(
        self,
        img: np.ndarray,
        lower: int | None,
        upper: int | None 
    ) -> np.ndarray:

        img = img[lower:upper, ::self._down_sample, ::self._down_sample]
        img = self._normalise(img)

        return img

    def _normalise(self, img: np.ndarray) -> np.ndarray:
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)
        img = 2 * img - 1
        return img
    
    def un_normalise(self, img: np.ndarray) -> np.ndarray:
        img = (img + 1) / 2
        img = img * (HU_MAX - HU_MIN) + HU_MIN
        return img

    @abstractmethod
    def data_generator(self):
        pass

    @abstractmethod
    def inference_generator(self):
        pass

    @property
    def example_images(self) -> dict[str, np.ndarray]:
        """ Return example images """

        return {"target": self._ex_targets, "source": self._ex_sources}

    @property
    def data(self) -> dict[str, dict[str, Path]]:
        """ Return list of all images """

        return {"target": self._targets, "source": self._sources}
