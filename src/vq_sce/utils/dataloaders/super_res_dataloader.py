import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from vq_sce import LQ_DEPTH, LQ_SLICE_THICK, RANDOM_SEED
from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader, DataDictType

# ----------------------------------------------------------------------------------------------------------------------------------------------------


class SuperResDataloader(BaseDataloader):
    """Super-resolution dataloader class."""

    _source_target_map: dict[str, str]

    def __init__(
        self,
        config: dict[str, Any],
        dataset_type: str,
        dev: bool = False,
    ) -> None:
        self.N = 10 if dev else None
        self._img_path = Path(config["data_path"])
        self._source_path = self._img_path / "LQ"
        self._target_path = self._img_path / "HQ"
        self._dataset_type = dataset_type
        self._config = config
        self._down_sample = config["down_sample"]

        with open(self._img_path / "source_coords.json") as fp:
            self._source_coords = json.load(fp)

        self._sources = {s.stem: s for s in self._source_path.glob("*")}
        self._targets = {}
        self._source_target_map = {}

        for s in self._sources.keys():
            candidate_target_paths = list(self._target_path.glob(f"{s[0:6]}*.npy"))
            candidate_targets = [c.stem for c in candidate_target_paths]

            sort_by_closest = sorted(
                candidate_targets,
                key=lambda x: abs(int(x[-3:]) - int(s[-3:])),
            )
            self._targets[sort_by_closest[0]] = (
                self._target_path / f"{sort_by_closest[0]}.npy"
            )
            self._source_target_map[s] = sort_by_closest[0]

        np.random.seed(RANDOM_SEED)
        self._train_val_split()
        self._generate_example_images()
        np.random.seed()

        self._source_ids = list(self._sources.keys())

    def _calc_coords(
        self,
        source_id: str,
        target_id: str,
    ) -> tuple[list[int], list[int]]:
        source_coords = list(self._source_coords[source_id].values())[0]
        target_coords = list(self._source_coords[target_id].values())[0]
        target_coords = [
            source_coords[0] - target_coords[0],
            source_coords[0] - target_coords[0] + (LQ_SLICE_THICK * LQ_DEPTH),
        ]

        return source_coords, target_coords

    def _generate_example_images(self) -> None:
        """Generate example images for saving each epoch."""
        ex_sources_ids = np.random.choice(
            list(self._sources.keys()),
            self._config["num_examples"],
        )

        ex_sources: list[npt.NDArray[np.float16]] = []
        ex_targets: list[npt.NDArray[np.float16]] = []

        for source_id in ex_sources_ids:
            target_id = self._source_target_map[source_id]
            _, hq_coords = self._calc_coords(source_id, target_id)

            target = np.load(self._target_path / f"{target_id}.npy").astype("float32")
            target = self._preprocess_image(target, hq_coords[0], hq_coords[1])
            source = np.load(self._source_path / f"{source_id}.npy").astype("float32")
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
        if self._dataset_type == "training":
            np.random.shuffle(self._source_ids)

        for source_id in self._source_ids[0 : self.N]:
            source = np.load(self._source_path / f"{source_id}.npy").astype("float32")
            source = self._preprocess_image(source, None, None)

            target_id = self._source_target_map[source_id]
            _, hq_coords = self._calc_coords(source_id, target_id)

            target = np.load(self._target_path / f"{target_id}.npy").astype("float32")
            target = self._preprocess_image(target, hq_coords[0], hq_coords[1])

            data_dict: DataDictType = {
                "source": source[:, :, :, np.newaxis],
                "target": target[:, :, :, np.newaxis],
            }

            yield data_dict

    def inference_generator(self) -> Iterator[dict[str, Any]]:
        for source_id in self._source_ids:
            source = np.load(self._source_path / f"{source_id}.npy").astype("float32")
            source = self._preprocess_image(source, None, None)

            target_id = self._source_target_map[source_id]
            _, hq_coords = self._calc_coords(source_id, target_id)

            target = np.load(self._target_path / f"{target_id}.npy").astype("float32")
            target = self._preprocess_image(target, hq_coords[0], hq_coords[1])

            yield {
                "source": source,
                "target": target,
                "source_id": source_id,
                "target_id": target_id,
            }


# ----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """.Routine for visually testing dataloader."""
    import matplotlib.pyplot as plt
    import yaml

    test_config = yaml.load(
        open(Path("src/vq_sce/utils/test_config.yml")),
        Loader=yaml.FullLoader,
    )

    TestLoader = SuperResDataloader(
        config=test_config["data"],
        dataset_type="training",
        dev=False,
    )

    output_types = ["source", "target"]

    train_ds = tf.data.Dataset.from_generator(
        TestLoader.data_generator,
        output_types={k: "float32" for k in output_types},
    )

    for data in train_ds.batch(2).take(16):
        source = TestLoader.un_normalise(data["source"])
        target = TestLoader.un_normalise(data["target"])

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, 1, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, 1, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, 6, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, 6, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 5)
        plt.imshow(
            target[0, 6, :, :, 0] - source[0, 1, :, :, 0],
            cmap="gray",
            vmin=-150,
            vmax=250,
        )
        plt.axis("off")

        plt.subplot(3, 2, 6)
        plt.imshow(
            target[1, 6, :, :, 0] - source[1, 1, :, :, 0],
            cmap="gray",
            vmin=-150,
            vmax=250,
        )
        plt.axis("off")

        plt.show()

    data = TestLoader.example_images

    source = TestLoader.un_normalise(data["source"])
    target = TestLoader.un_normalise(data["target"])

    fig, axs = plt.subplots(target.shape[0], 2)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, 1, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target[i, 6, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")

    plt.show()
