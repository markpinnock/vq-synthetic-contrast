import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

from vq_sce.utils.patch_utils import generate_indices, extract_patches


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ImgLoader class: data_generator method for use with tf.data.Dataset.from_generator """

class SuperResDataloader:
    def __init__(self, config: dict, dataset_type: str):
        # Expects at least two sub-folders within data folder e.g. "AC", "VC, "HQ"
        self.img_path = Path(config["data_path"]) / "Images"
        self.seg_path = Path(config["data_path"]) / "Segmentations"
        self._dataset_type = dataset_type
        self.config = config
        self.down_sample = config["down_sample"]
        self.num_targets = len(config["target"])
        self._patch_size = config["patch_size"]
        if config["times"] is not None:
            self._json = json.load(open(Path(config["data_path"]) / config["times"], 'r'))
        else:
            self._json = None

        self.param_1 = self.config["norm_param_1"]
        self.param_2 = self.config["norm_param_2"]

        # Optional list of targets and sources e.g. ["AC", "VC"], ["HQ"]
        self._targets = []
        self._sources = []
        self._segs = []

        if len(config["target"]) > 0:
            for key in config["target"]:
                self._targets += [t for t in os.listdir(self.img_path) if key in t]

        elif len(config["target"]) == 0:
            self._targets += os.listdir(self.img_path)

        if len(config["source"]) > 0:
            for key in config["source"]:
                self._sources += [s for s in os.listdir(self.img_path) if key in s]

        elif len(config["source"]) == 0:
            self._sources += os.listdir(self.img_path)
        
        if len(config["segs"]) > 0:
            self._segs += os.listdir(self.seg_path)

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources, {len(self._segs)} segmentations")
        print(f"Using unpaired loader for {self._dataset_type}")

        self.train_val_split()

        if len(self.config["target"]) > 0:
            self._subject_targets = {k: [img for img in v if img[6:8] in self.config["target"]] for k, v in self._subject_imgs.items()}
        else:
            self._subject_targets = None

    def example_images(self):
        if self._json is not None:
            return {
                "source": self._normalise(self._ex_sources),
                "target": self._normalise(self._ex_targets),
                "times": self._ex_target_times
                }
        else:
            return {
                "source": self._normalise(self._ex_sources),
                "target": self._normalise(self._ex_targets)
                }
    
    def train_val_split(self, seed: int = 5) -> None:
        # Get unique subject IDs for subject-level train/val split
        self._unique_ids = []

        for img_id in self._targets:
            if img_id[0:4] not in self._unique_ids:
                self._unique_ids.append(img_id[0:4])

        self._unique_ids.sort()
        self._subject_imgs = {}

        # Need procedure IDs (as poss. >1 per subject) to build ordered index of subjects' images
        self._subject_imgs = {}

        for img_id in self._targets + self._sources:
            if img_id[0:6] not in self._subject_imgs.keys():
                self._subject_imgs[img_id[0:6]] = []

            if img_id[:-4] not in self._subject_imgs[img_id[0:6]]:
                self._subject_imgs[img_id[0:6]].append(img_id[:-4])

        for key in self._subject_imgs.keys():
            self._subject_imgs[key] = sorted(self._subject_imgs[key], key=lambda x: int(x[-3:]))

        if self.config["fold"] > self.config["cv_folds"] - 1:
            raise ValueError(f"Fold number {self.config['fold']} of {self.config['cv_folds']} folds")

        np.random.seed(seed)
        N = len(self._unique_ids)
        np.random.shuffle(self._unique_ids)

        # Split into folds by subject
        if self.config["cv_folds"] > 1:
            if seed == None:
                self._unique_ids.sort()

            num_in_fold = N // self.config["cv_folds"]

            if self._dataset_type == "training":
                fold_ids = self._unique_ids[0:self.config["fold"] * num_in_fold] + self._unique_ids[(self.config["fold"] + 1) * num_in_fold:]
            elif self._dataset_type == "validation":
                fold_ids = self._unique_ids[self.config["fold"] * num_in_fold:(self.config["fold"] + 1) * num_in_fold]
            else:
                raise ValueError("Select 'training' or 'validation'")

            self._fold_targets = []
            self._fold_sources = []
            self._fold_targets = sorted([img for img in self._targets if img[0:4] in fold_ids])
            self._fold_sources = sorted([img for img in self._sources if img[0:4] in fold_ids])
            self._fold_segs = sorted([seg for seg in self._segs if seg[0:4] in fold_ids])
        
        elif self.config["cv_folds"] == 1:
            self._fold_targets = self._targets
            self._fold_sources = self._sources
            self._fold_segs = self._segs
        
        else:
            raise ValueError("Number of folds must be > 0")

        example_idx = np.random.randint(0, len(self._fold_sources), self.config["num_examples"])
        ex_sources_list = list(np.array([self._fold_sources]).squeeze()[example_idx])

        try:
            ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'AC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list) // 2]]
            ex_targets_list += [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'VC' in t and t not in s]) for s in ex_sources_list[len(ex_sources_list) // 2:]]
        except ValueError:
            try:
                ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'AC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list)]]
            except ValueError:
                ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'VC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list)]]

        ex_sources = [np.load(Path(self.img_path) / img)[::self.down_sample, ::self.down_sample, :] for img in ex_sources_list]
        ex_targets = [np.load(Path(self.img_path) / img)[::self.down_sample, ::self.down_sample, :] for img in ex_targets_list]
        ex_sources_stack = []
        ex_targets_stack = []

        for source, target in zip(ex_sources, ex_targets):
            sub_source = source[:, :, (source.shape[2] // 3):(source.shape[2] // 3 + self._patch_size[2])]
            sub_target = target[:, :, (target.shape[2] // 3):(target.shape[2] // 3 + self._patch_size[2])]

            if sub_source.shape[2] < self._patch_size[2]:
                sub_source = source[:, :, -self._patch_size[2]:]
                sub_target = target[:, :, -self._patch_size[2]:]

            ex_sources_stack.append(sub_source)
            ex_targets_stack.append(sub_target)

        self._ex_sources = np.stack(ex_sources_stack, axis=0)[:, :, :, :, np.newaxis].astype("float32")
        self._ex_targets = np.stack(ex_targets_stack, axis=0)[:, :, :, :, np.newaxis].astype("float32")

        if self._json is not None:
            self._ex_source_times = np.stack([self._json[name[:-4] + ".nrrd"] for name in ex_sources_list], axis=0).astype("float32")
            self._ex_target_times = np.stack([self._json[name[:-4] + ".nrrd"] for name in ex_targets_list], axis=0).astype("float32")

        else:
            self._ex_source_times = []
            self._ex_target_times = []

        np.random.seed()
        
        print(f"{len(self._fold_targets)} of {len(self._targets)} examples in {self._dataset_type} folds")

    @property
    def unique_ids(self) -> list:
        return self._unique_ids
    
    @property
    def data(self) -> dict:
        """ Return list of all images """
        return {"targets": self._targets, "sources": self._sources}
    
    @property
    def fold_data(self) -> dict:
        """ Return list of all images in training or validation fold """
        return {"targets": self._fold_targets, "sources": self._fold_sources}
    
    @property
    def subject_imgs(self):
        """ Return list of images indexed by procedure """
        return self._subject_imgs
    
    def img_pairer(self, source: object, direction: str = None) -> dict:
        # TODO: add forwards/backwards sampling, return idx
        if self._subject_targets is None:
            target_candidates = list(self._subject_imgs[source[0:6]])
        else:
            target_candidates = list(self._subject_targets[source[0:6]])

        try:
            target_candidates.remove(source[0:-4])
        except ValueError:
            pass

        # target_stem = target_candidates[np.random.randint(len(target_candidates))]
        target = [f"{t}.npy" for t in target_candidates]

        return {"target": target, "source": source}
    
    def _normalise(self, img):
        return (img - self.param_1) / (self.param_2 - self.param_1)
    
    def un_normalise(self, img):
        return img * (self.param_2 - self.param_1) + self.param_1

    def data_generator(self):
        if self._dataset_type == "training":
            np.random.shuffle(self._fold_sources)

        N = len(self._fold_sources)
        i = 0

        # Pair source and target images
        while i < N:
            source_name = self._fold_sources[i]
            names = self.img_pairer(source_name)
            source_name = names["source"]
            source = np.load(Path(self.img_path) / source_name)[::self.down_sample, ::self.down_sample, :]
            source = self._normalise(source)

            for target_name in names["target"]:
                target = np.load(Path(self.img_path) / target_name)[::self.down_sample, ::self.down_sample, :]
                target = self._normalise(target)

                total_depth = target.shape[2]
                num_iter = total_depth // self._patch_size[2]

                # TODO: allow using different seg channels
                if len(self._fold_segs) > 0:
                    candidate_segs = glob.glob(str(Path(self.seg_path) / f"{target_name[0:6]}AC*{target_name[-4:]}"))
                    assert len(candidate_segs) == 1, candidate_segs
                    seg = np.load(candidate_segs[0])[::self.down_sample, ::self.down_sample, :]
                    seg[seg > 1] = 1
                else:
                    seg = None

                for _ in range(num_iter):
                    z = np.random.randint(0, total_depth - self._patch_size[2] + 1)
                    sub_target = target[:, :, z:(z + self._patch_size[2]), np.newaxis]
                    sub_source = source[:, :, z:(z + self._patch_size[2]), np.newaxis]

                    data_dict = {
                        "source": sub_source,
                        "target": sub_target
                    }

                    # TODO: allow using different seg channels
                    if seg is not None:
                        sub_seg = seg[:, :, z:(z + self._patch_size[2]), np.newaxis]
                        data_dict["seg"] = sub_seg
                        # TODO: return index

                    if self._json is not None:
                        data_dict["times"] = self._json[target_name[:-4] + ".nrrd"]

                    yield data_dict

            i += 1

    def inference_generator(self):
        N = len(self._fold_sources)
        i = 0

        # Pair source and target images
        while i < N:
            source_name = self._fold_sources[i]

            if len(self.sub_folders) == 0:
                source = np.load(Path(self._img_paths) / source_name)
            else:
                source = np.load(Path(self._img_paths[source_name[6:8]]) / source_name)

            source = self._normalise(source)
            patches, indices = extract_patches(source, self.config["xy_patch"], self.config["stride_length"], self._patch_size, self.down_sample)

            for patch, index in zip(patches, indices):
                yield {"source": patch, "subject_ID": source_name, "x": index[0], "y": index[1], "z": index[2]}

            i += 1

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
        linear_coords = generate_indices(source, self.config["stride_length"], self._patch_size, self.down_sample)

        source = self._normalise(source)
        linear_source = tf.reshape(source, -1)

        for coords in linear_coords:
            patch = tf.reshape(tf.gather(linear_source, coords), self._patch_size + [1])

            yield {"source": patch, "subject_ID": source_name, "coords": coords}


#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":
    import yaml

    """ Routine for visually testing dataloader """

    test_config = yaml.load(open(Path("vec_quant_sCE/utils/test_config.yml"), 'r'), Loader=yaml.FullLoader)

    TestLoader = SuperResDataloader(config=test_config["data"], dataset_type="training")

    output_types = ["source", "target"]

    if len(test_config["data"]["segs"]) > 0:
        output_types += ["seg"]
    
    if test_config["data"]["times"] is not None:
        output_types += ["times"]
    
    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types={k: "float32" for k in output_types})

    for data in train_ds.batch(2).take(16):
        source = TestLoader.un_normalise(data["source"])
        target = TestLoader.un_normalise(data["target"])

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["times"][0].numpy())

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["times"][1].numpy())

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["times"][0].numpy())

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["times"][1].numpy())

        if "seg" in data.keys():
            plt.subplot(3, 2, 5)
            plt.imshow(data["seg"][0, :, :, 0, 0].numpy())
            plt.axis("off")
            plt.subplot(3, 2, 6)
            plt.imshow(data["seg"][1, :, :, 0, 0].numpy())
            plt.axis("off")

        plt.show()

    data = TestLoader.example_images()

    source = TestLoader.un_normalise(data["source"])
    target = TestLoader.un_normalise(data["target"])
    
    fig, axs = plt.subplots(target.shape[0], 2)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")

        if "times" in data.keys():
            axs[i, 1].set_title(data["times"][i])
    
    plt.show()
