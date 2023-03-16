import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Any, Iterator

from vq_sce import RANDOM_SEED, MIN_HQ_DEPTH
from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader, DataDictType


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ContrastDataloader class:
    data_generator method for use with tf.data.Dataset.from_generator
"""

class ContrastDataloader(BaseDataloader):
    _target_source_map: dict[str, list[str]]

    def __init__(self, config: dict[str, Any], dataset_type: str, dev: bool = False) -> None:

        self.N = 10 if dev else None
        self._img_path = Path(config["data_path"])
        self._target_path = self._img_path / "CE"
        self._source_path = self._img_path / "HQ"
        self._dataset_type = dataset_type
        self._config = config
        self._down_sample = config["down_sample"]

        with open(self._img_path / "source_coords.json", 'r') as fp:
            self._source_coords = json.load(fp)

        self._sources = {}
        self._targets = {t.stem: t for t in self._target_path.glob("*")}

        for t in self._targets.keys():
            source_ids = list(self._source_coords[t].keys())
            for source_id in source_ids:
                candidate_sources = list(self._source_path.glob(f"{source_id}.npy"))
                assert len(candidate_sources) == 1, f"Too many sources: {candidate_sources}"
                self._sources[candidate_sources[0].stem] = candidate_sources[0]

        np.random.seed(RANDOM_SEED)
        self._train_val_split()
        self._generate_example_images()
        np.random.seed()

        self._target_ids = list(self._targets.keys())
        self._target_source_map = {
            k: list(self._source_coords[k].keys()) for k in self._target_ids
        }

    def _calc_coords(
        self,
        target_id: str
    ) -> list[int]:

        target_coords: list[int] = list(self._source_coords[target_id].values())[0]
        return target_coords

    def _generate_example_images(self) -> None:
        """ Generate example images for saving each epoch """

        ex_sources_ids = []
        ex_targets_ids = np.random.choice(list(self._targets.keys()), self._config["num_examples"])
        for t in ex_targets_ids:
            source_id = np.random.choice(list(self._source_coords[t].keys()), 1)
            ex_sources_ids.append(source_id[0])

        ex_sources = []
        ex_targets = []
        for target_id, source_id in zip(ex_targets_ids, ex_sources_ids):
            ce_coords = self._calc_coords(target_id)

            target = np.load(self._target_path / f"{target_id}.npy")
            target = self._preprocess_image(target, None, None)
            source = np.load(self._source_path / f"{source_id}.npy")
            source = self._preprocess_image(source, ce_coords[0], ce_coords[1])

            lower = target.shape[0] // 3
            upper = target.shape[0] // 3 + MIN_HQ_DEPTH
            if upper > target.shape[0]:
                lower = None
                upper = -MIN_HQ_DEPTH

            ex_targets.append(target[lower:upper, :, :])
            ex_sources.append(source[lower:upper, :, :])

        self._ex_sources = np.stack(ex_sources, axis=0) \
            [:, :, :, :, np.newaxis].astype("float32")
        self._ex_targets = np.stack(ex_targets, axis=0) \
            [:, :, :, :, np.newaxis].astype("float32")

    def data_generator(self) -> Iterator[DataDictType]:
        if self._dataset_type == "training":
            np.random.shuffle(self._target_ids)

        for target_id in self._target_ids[0:self.N]:
            target = np.load(self._target_path / f"{target_id}.npy")
            target = self._preprocess_image(target, None, None)

            for source_id in self._target_source_map[target_id]:
                ce_coords = self._calc_coords(target_id)
                source = np.load(self._source_path / f"{source_id}.npy")
                source = self._preprocess_image(source, ce_coords[0], ce_coords[1])
                total_depth = target.shape[0]
                num_iter = total_depth // MIN_HQ_DEPTH

                for _ in range(num_iter):
                    z = np.random.randint(0, total_depth - MIN_HQ_DEPTH + 1)
                    sub_target = target[z:(z + MIN_HQ_DEPTH), :, :, np.newaxis]
                    sub_source = source[z:(z + MIN_HQ_DEPTH), :, :, np.newaxis]

                    data_dict: DataDictType = {
                        "source": sub_source,
                        "target": sub_target
                    }
    
                    yield data_dict

    def inference_generator(self) -> Iterator[dict[str, Any]]:
        for target_id in self._target_ids:
            for source_id in self._target_source_map[target_id]:
                lower, upper = self._source_coords[target_id][source_id]
                source = np.load(self._source_path / f"{source_id}.npy")
                source = self._preprocess_image(source, lower, upper)

                yield {"source": source, "subject_id": source_id}

    def subject_generator(self, source_name: Any) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import yaml

    """ Routine for visually testing dataloader """

    test_config = yaml.load(open(Path("src/vq_sce/utils/test_config.yml"), 'r'), Loader=yaml.FullLoader)

    TestLoader = ContrastDataloader(config=test_config["data"], dataset_type="training", dev=False)

    output_types = ["source", "target"]

    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types={k: "float32" for k in output_types})

    for data in train_ds.batch(2).take(4):
        source = TestLoader.un_normalise(data["source"])
        target = TestLoader.un_normalise(data["target"])

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.subplot(3, 2, 5)
        plt.imshow(target[0, 11, :, :, 0].numpy() - source[0, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.subplot(3, 2, 6)
        plt.imshow(target[1, 11, :, :, 0].numpy() - source[1, 11, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)  # type: ignore[attr-defined]
        plt.axis("off")

        plt.show()

    data = TestLoader.example_images

    source = TestLoader.un_normalise(data["source"])
    target = TestLoader.un_normalise(data["target"])
    
    fig, axs = plt.subplots(target.shape[0], 2)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, 11, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target[i, 11, :, :, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")
    
    plt.show()
