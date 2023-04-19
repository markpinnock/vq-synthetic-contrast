import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from vq_sce import LQ_DEPTH, LQ_SLICE_THICK, RANDOM_SEED
from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader, DataDictType


class JointDataset(BaseDataloader):
    def __init__(self, config: dict[str, Any], dataset_type: str) -> None:
        self._img_path = Path(config["data_path"])
        self._target_path = self._img_path / "CE"
        self._source_path = self._img_path / "LQ"
        self._dataset_type = dataset_type
        self._config = config
        self._down_sample = config["down_sample"]

        with open(self._img_path / "source_coords.json") as fp:
            self._source_coords = json.load(fp)

        self._sources = {}
        self._targets = {t.stem: t for t in self._target_path.glob("*")}
        self._source_target_map = {}

        for t in self._targets.keys():
            candidate_sources = list(self._source_path.glob(f"{t[0:6]}*.npy"))
            if len(candidate_sources) == 0:
                continue
            else:
                self._sources[candidate_sources[-1].stem] = candidate_sources[-1]
                self._source_target_map[candidate_sources[-1].stem] = t

        np.random.seed(RANDOM_SEED)
        self._train_val_split()
        self._generate_example_images()
        np.random.seed()

    def _calc_coords(
        self,
        source_id: str,
        target_id: str,
    ) -> tuple[list[int], list[int]]:
        lq_coords = list(self._source_coords[source_id].values())[0]
        ce_coords = list(self._source_coords[target_id].values())[0]
        # ce_level = lq_coords[0] - ce_coords[0]

        ce_coords = [
            lq_coords[0] - ce_coords[0],
            lq_coords[0] - ce_coords[0] + (LQ_SLICE_THICK * LQ_DEPTH),
        ]

        return lq_coords, ce_coords

    def _generate_example_images(self) -> None:
        """Generate example images for saving each epoch."""
        ex_sources_ids = np.random.choice(
            list(self._sources.keys()),
            self._config["num_examples"],
        )

        ex_sources = []
        ex_targets = []

        for source_id in ex_sources_ids:
            target_id = self._source_target_map[source_id]
            _, ce_coords = self._calc_coords(source_id, target_id)

            target = np.load(self._target_path / f"{target_id}.npy")
            target = self._preprocess_image(target, ce_coords[0], ce_coords[1])
            source = np.load(self._source_path / f"{source_id}.npy")
            source = self._preprocess_image(source, None, None)

            ex_targets.append(target)
            ex_sources.append(source)

        self._ex_sources = np.stack(ex_sources, axis=0)[:, :, :, :, np.newaxis].astype(
            "float32",
        )
        self._ex_targets = np.stack(ex_targets, axis=0)[:, :, :, :, np.newaxis].astype(
            "float32",
        )

    def data_generator(self) -> Iterator[DataDictType]:
        raise NotImplementedError

    def inference_generator(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """Routine for visually testing dataloader."""
    import matplotlib.pyplot as plt
    import yaml

    test_config = yaml.load(
        open(Path("src/vq_sce/utils/test_config.yml")),
        Loader=yaml.FullLoader,
    )

    TestLoader = JointDataset(config=test_config["data"], dataset_type="training")
    data = TestLoader.example_images

    source = TestLoader.un_normalise(data["source"])
    target = TestLoader.un_normalise(data["target"])

    fig, axs = plt.subplots(target.shape[0], 2)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, 2, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target[i, 11, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")

    plt.show()
