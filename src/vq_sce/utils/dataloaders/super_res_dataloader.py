import glob
import json
import numpy as np
from pathlib import Path
import tensorflow as tf

from vq_sce import RANDOM_SEED
from vq_sce.utils.dataloaders.base_dataloader import BaseDataloader
from vq_sce.utils.patch_utils import generate_indices, extract_patches

#----------------------------------------------------------------------------------------------------------------------------------------------------
""" SuperResDataloader class:
    data_generator method for use with tf.data.Dataset.from_generator
"""

class SuperResDataloader(BaseDataloader):
    def __init__(self, config: dict, dataset_type: str) -> None:

        self._img_path = Path(config["data_path"])
        self._source_path = self._img_path / "LQ"
        self._target_path = self._img_path / "HQ"
        self._dataset_type = dataset_type
        self._config = config
        self._down_sample = config["down_sample"]
        self._patch_size = config["patch_size"]

        with open(self._img_path / "source_coords.json", 'r') as fp:
            self._source_coords = json.load(fp)

        self._sources = {s.stem: s for s in self._source_path.glob("*")}
        self._targets = {}

        for s in self._sources.keys():
            target_ids = list(self._source_coords[s].keys())
            for target_id in target_ids:
                candidate_targets = list(self._target_path.glob(f"{target_id}.npy"))
                assert len(candidate_targets) == 1, f"Too many sources: {candidate_targets}"
                self._targets[candidate_targets[0].stem] = candidate_targets[0]

        np.random.seed(RANDOM_SEED)
        self._train_val_split()
        self._generate_example_images()
        np.random.seed()

        self._source_ids = list(self._sources.keys())
        self._source_target_map = {
            k: list(self._source_coords[k].keys()) for k in self._source_ids
        }
    
    def _generate_example_images(self) -> None:
        """ Generate example images for saving each epoch """

        ex_targets_ids = []
        ex_sources_ids = np.random.choice(list(self._sources.keys()), self._config["num_examples"])
        for s in ex_sources_ids:
            target_id = np.random.choice(list(self._source_coords[s].keys()), 1)
            ex_targets_ids.append(target_id[0])

        ex_sources = []
        ex_targets = []
        for target_id, source_id in zip(ex_targets_ids, ex_sources_ids):
            lower, upper = self._source_coords[source_id][target_id]

            target = np.load(self._target_path / f"{target_id}.npy")
            target = self._preprocess_image(target, lower, upper)
            source = np.load(self._source_path / f"{source_id}.npy")
            source = self._preprocess_image(source, None, None)

            ex_targets.append(target)
            ex_sources.append(source)

        self._ex_sources = np.stack(ex_sources, axis=0) \
            [:, :, :, :, np.newaxis].astype("float32")
        self._ex_targets = np.stack(ex_targets, axis=0) \
            [:, :, :, :, np.newaxis].astype("float32")

    def data_generator(self):
        if self._dataset_type == "training":
            np.random.shuffle(self._source_ids)

        for source_id in self._source_ids:
            source = np.load(self._source_path / f"{source_id}.npy")
            source = self._preprocess_image(source, None, None)

            for target_id in self._source_target_map[source_id]:
                lower, upper = self._source_coords[source_id][target_id]
                target = np.load(self._target_path / f"{target_id}.npy")
                target = self._preprocess_image(target, lower, upper)

                data_dict = {
                    "source": source[:, :, :, np.newaxis],
                    "target": target[:, :, :, np.newaxis]
                }

                yield data_dict

    def inference_generator(self):
        for source_id in self._source_ids:
            source = np.load(self._source_path / f"{source_id}.npy")
            source = self._preprocess_image(source, None, None)

            yield {"source": source, "subject_id": source_id}

    def subject_generator(self, source_name):
        source_name = source_name.decode("utf-8")

        if len(self.sub_folders) == 0:
            source = np.load(Path(self._img_paths) / source_name)
        else:
            source = np.load(Path(self._img_paths[source_name[6:8]]) / source_name)

        # Linear coords are what we'll use to do our patch updates in 1D
        # E.g. [1, 2, 3
        #       4, 5, 6
        #       7, 8, 9]
        linear_coords = generate_indices(source, self._config["stride_length"], self._patch_size, self.down_sample)

        source = self._normalise(source)
        linear_source = tf.reshape(source, -1)

        for coords in linear_coords:
            patch = tf.reshape(tf.gather(linear_source, coords), self._patch_size + [1])

            yield {"source": patch, "subject_ID": source_name, "coords": coords}


#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import yaml

    """ Routine for visually testing dataloader """

    test_config = yaml.load(open(Path("src/vq_sce/utils/test_config.yml"), 'r'), Loader=yaml.FullLoader)

    TestLoader = SuperResDataloader(config=test_config["data"], dataset_type="training")

    output_types = ["source", "target"]
    
    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types={k: "float32" for k in output_types})

    for data in train_ds.batch(2).take(16):
        source = TestLoader.un_normalise(data["source"])
        target = TestLoader.un_normalise(data["target"])
        print(source.shape, target.shape)
        plt.subplot(3, 2, 1)
        plt.imshow(source[0, 1, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, 1, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, 6, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, 6, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 5)
        plt.imshow(target[0, 6, :, :, 0].numpy() - source[0, 1, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(3, 2, 6)
        plt.imshow(target[1, 6, :, :, 0].numpy() - source[1, 1, :, :, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
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
