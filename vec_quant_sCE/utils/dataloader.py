import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from abc import ABC, abstractmethod

from syntheticcontrast_v02.utils.patch_utils import generate_indices, extract_patches


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ImgLoader class: data_generator method for use with tf.data.Dataset.from_generator """

class BaseImgLoader(ABC):
    def __init__(self, config: dict, dataset_type: str):
        # Expects at least two sub-folders within data folder e.g. "AC", "VC, "HQ"
        img_path = f"{config['data_path']}/Images"
        seg_path = f"{config['data_path']}/Segmentations"
        self.sub_folders = [f for f in os.listdir(img_path) if os.path.isdir(f"{img_path}/{f}")]
        self.seg_folders = [f for f in os.listdir(seg_path) if os.path.isdir(f"{img_path}/{f}")]

        if len(self.sub_folders) == 0:
            print("==================================================")
            print("Assuming unpaired dataset")
            self._img_paths = img_path
            self._seg_paths = seg_path

        else:
            self._img_paths = {key: f"{img_path}/{key}" for key in self.sub_folders}
            self._seg_paths = {key: f"{seg_path}/{key}" for key in self.seg_folders}

        self._dataset_type = dataset_type
        self.config = config
        self.down_sample = config["down_sample"]
        self.num_targets = len(config["target"])
        self._patch_size = config["patch_size"]

        if config["times"] is not None:
            self._json = json.load(open(f"{config['data_path']}/{config['times']}", 'r'))
        
        else:
            self._json = None

    def example_images(self):
        if self._json is not None:
            return {
                "real_source": self._normalise(self._ex_sources),
                "real_target": self._normalise(self._ex_targets),
                "source_times": self._ex_source_times,
                "target_times": self._ex_target_times
                }
        else:
            return {
                "real_source": self._normalise(self._ex_sources),
                "real_target": self._normalise(self._ex_targets)
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

        if len(self.sub_folders) == 0:
            try:
                ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'AC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list) // 2]]
                ex_targets_list += [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'VC' in t and t not in s]) for s in ex_sources_list[len(ex_sources_list) // 2:]]
            except ValueError:
                try:
                    ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'AC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list)]]
                except ValueError:
                    ex_targets_list = [np.random.choice([t for t in self._fold_targets if s[0:6] in t and 'VC' in t and t not in s]) for s in ex_sources_list[0:len(ex_sources_list)]]

            ex_sources = [np.load(f"{self._img_paths}/{img}") for img in ex_sources_list]
            ex_targets = [np.load(f"{self._img_paths}/{img}") for img in ex_targets_list]

        else:
            ex_targets_list = list(np.array([self._fold_targets]).squeeze()[example_idx])
            ex_sources = [np.load(f"{self._img_paths[img[6:8]]}/{img}") for img in ex_sources_list]
            ex_targets = [np.load(f"{self._img_paths[img[6:8]]}/{img}") for img in ex_targets_list]
          
        ex_sources_stack = []
        ex_targets_stack = []
        mid_x = ex_sources[0].shape[0] // 2
        mid_y = ex_sources[0].shape[1] // 2

        for source, target in zip(ex_sources, ex_targets):
            sub_source = source[(mid_x - self._patch_size[0] // 2):(mid_x + self._patch_size[0] // 2), (mid_y - self._patch_size[1] // 2):(mid_y + self._patch_size[1] // 2), (source.shape[2] // 3):(source.shape[2] // 3 + self._patch_size[2])]
            sub_target = target[(mid_x - self._patch_size[0] // 2):(mid_x + self._patch_size[0] // 2), (mid_y - self._patch_size[1] // 2):(mid_y + self._patch_size[1] // 2), (target.shape[2] // 3):(target.shape[2] // 3 + self._patch_size[2])]

            if sub_source.shape[2] < self._patch_size[2]:
                sub_source = source[(mid_x - self._patch_size[0] // 2):(mid_x + self._patch_size[0] // 2), (mid_y - self._patch_size[1] // 2):(mid_y + self._patch_size[1] // 2), -self._patch_size[2]:]
                sub_target = target[(mid_x - self._patch_size[0] // 2):(mid_x + self._patch_size[0] // 2), (mid_y - self._patch_size[1] // 2):(mid_y + self._patch_size[1] // 2), -self._patch_size[2]:]

            ex_sources_stack.append(sub_source)
            ex_targets_stack.append(sub_target)

        self._ex_sources = np.stack(ex_sources_stack, axis=0)
        self._ex_targets = np.stack(ex_targets_stack, axis=0)
        self._ex_sources = self._ex_sources[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")
        self._ex_targets = self._ex_targets[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")

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
        raise NotImplementedError
    
    def set_normalisation(self, param_1: float = None, param_2: float = None):
        # Mean -281.528, std = 261.552
        # Min -500, max = 22451
        self.norm_type = self.config["norm_type"]

        # Override if custom parameters passed
        if param_1 is not None and param_2 is not None:
            self.param_1 = param_1
            self.param_2 = param_2

        # Otherwise, use parameters provided in config yaml
        elif self.config["norm_param_1"] is not None and self.config["norm_param_1"] is not None:
            self.param_1 = self.config["norm_param_1"]
            self.param_2 = self.config["norm_param_2"]

        # Otherwise, calculate parameters
        else:
            # If mean and std of data not available, we get rolling averages
            if self.norm_type == "meanstd" or self.norm_type == "std":
                mean = 0
                std = 0

                for img in self._targets + self._sources:
                    im = np.load(f"{self._img_paths[img[6:8]]}/{img}")
                    mean = 0.99 * mean + 0.01 * im.mean()
                    std = 0.99 * std + 0.01 * im.std()
                
                self.param_1 = mean
                self.param_2 = std

            # If min and max not available, we get min and max of whole dataset
            elif self.norm_type == "minmax":
                min_val = 2048
                max_val = -2048

                for img in self._targets + self._sources:
                    im = np.load(f"{self._img_paths[img[6:8]]}/{img}")
                    min_val = np.min([min_val, im.min()])
                    max_val = np.max([max_val, im.max()])
                
                self.param_1 = min_val
                self.param_2 = max_val
        
            else:
                raise ValueError("Choose meanstd or minmax")

        print("==================================================")
        print(f"{self.norm_type} normalisation: mean/min {self.param_1}, std/max {self.param_2}")

        return self.param_1, self.param_2
    
    @property
    def norm_params(self):
        """ Return mean/std or min/max parameters """
        return (self.param_1, self.param_2)
    
    @abstractmethod
    def img_pairer(self):
        raise NotImplementedError
    
    def _normalise(self, img):
        if self.norm_type == "meanstd":
            return (img - self.param_1) / self.param_2
        elif self.norm_type == "std":
            return img / self.param_2
        else:
            return (img - self.param_1) / (self.param_2 - self.param_1)
    
    def un_normalise(self, img):
        if self.norm_type == "meanstd":
            return img * self.param_2 + self.param_1
        elif self.norm_type == "std":
            return img * self.param_2
        else:
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

            if len(self.sub_folders) == 0:
                source = np.load(f"{self._img_paths}/{source_name}")
            else:
                source = np.load(f"{self._img_paths[source_name[6:8]]}/{source_name}")

            for target_name in names["target"]:
                if len(self.sub_folders) == 0:
                    target = np.load(f"{self._img_paths}/{target_name}")
                else:
                    target = np.load(f"{self._img_paths[target_name[6:8]]}/{target_name}")

                # Extract patches
                total_height = target.shape[0]
                total_width = target.shape[1]
                total_depth = target.shape[2]
                num_iter = (total_height // self._patch_size[0]) * (total_width // self._patch_size[1]) * (total_depth // self._patch_size[2])

                for _ in range(num_iter):
                    x = np.random.randint(0, total_width - self._patch_size[0] + 1)
                    y = np.random.randint(0, total_width - self._patch_size[1] + 1)
                    z = np.random.randint(0, total_depth - self._patch_size[2] + 1)

                    sub_target = target[x:(x + self._patch_size[0]):self.down_sample, y:(y + self._patch_size[1]):self.down_sample, z:(z + self._patch_size[2]), np.newaxis]
                    sub_source = source[x:(x + self._patch_size[0]):self.down_sample, y:(y + self._patch_size[1]):self.down_sample, z:(z + self._patch_size[2]), np.newaxis]
                    sub_target = self._normalise(sub_target)
                    sub_source = self._normalise(sub_source)

                    if self._json is not None:
                        source_time = self._json[names["source"][:-4] + ".nrrd"]
                        target_time = self._json[target_name[:-4] + ".nrrd"]

                    # TODO: allow using different seg channels
                    if len(self._fold_segs) > 0:
                        if len(self.sub_folders) == 0:
                            candidate_segs = glob.glob(f"{self._seg_paths}/{target_name[0:6]}AC*{target_name[-4:]}")
                            assert len(candidate_segs) == 1, candidate_segs
                            seg = np.load(candidate_segs[0])
                            seg = seg[x:(x + self._patch_size[0]):self.down_sample, y:(y + self._patch_size[1]):self.down_sample, z:(z + self._patch_size[2]), np.newaxis]
                            seg[seg > 1] = 1
                            # TODO: return index

                        else:
                            seg = np.load(f"{self._seg_paths[target_name[6:8]]}/{target_name}")
                            seg = seg[x:(x + self._patch_size[0]):self.down_sample, y:(y + self._patch_size[1]):self.down_sample, z:(z + self._patch_size[2]), np.newaxis]
                            seg[seg > 1] = 1

                        if self._json is not None:
                            yield {
                                "real_source": sub_source,
                                "real_target": sub_target,
                                "seg": seg,
                                "source_times": source_time,
                                "target_times": target_time
                                }
                        else:
                            yield {
                                "real_source": sub_source,
                                "real_target": sub_target,
                                "seg": seg
                                }

                    else:
                        if self._json is not None:
                            yield {
                                "real_source": sub_source,
                                "real_target": sub_target,
                                "source_times": source_time,
                                "target_times": target_time
                                }
                        else:
                            yield {
                                "real_source": sub_source,
                                "real_target": sub_target
                                }

            i += 1

    def inference_generator(self):
        N = len(self._fold_sources)
        i = 0

        # Pair source and target images
        while i < N:
            source_name = self._fold_sources[i]

            if len(self.sub_folders) == 0:
                source = np.load(f"{self._img_paths}/{source_name}")
            else:
                source = np.load(f"{self._img_paths[source_name[6:8]]}/{source_name}")

            source = self._normalise(source)
            patches, indices = extract_patches(source, self.config["xy_patch"], self.config["stride_length"], self._patch_size, self.down_sample)

            for patch, index in zip(patches, indices):
                yield {"real_source": patch, "subject_ID": source_name, "x": index[0], "y": index[1], "z": index[2]}

            i += 1

    def subject_generator(self, source_name):
        source_name = source_name.decode("utf-8")

        if len(self.sub_folders) == 0:
            source = np.load(f"{self._img_paths}/{source_name}")
        else:
            source = np.load(f"{self._img_paths[source_name[6:8]]}/{source_name}")

        # Linear coords are what we'll use to do our patch updates in 1D
        # E.g. [1, 2, 3
        #       4, 5, 6
        #       7, 8, 9]
        linear_coords = generate_indices(source, self.config["stride_length"], self._patch_size, self.down_sample)

        source = self._normalise(source)
        linear_source = tf.reshape(source, -1)

        for coords in linear_coords:
            patch = tf.reshape(tf.gather(linear_source, coords), self._patch_size + [1])

            yield {"real_source": patch, "subject_ID": source_name, "coords": coords}


#-------------------------------------------------------------------------
""" Data loader for one to one source-target pairings """

class PairedLoader(BaseImgLoader):
    def __init__(self, config: dict, dataset_type: str):
        super().__init__(config, dataset_type)

        # Expects list of targets and sources e.g. ["AC", "VC"], ["HQ"]
        self._targets = []
        self._sources = []
        self._segs = []

        if len(config["target"]) > 0:
            for key in config["target"]:
                self._targets += os.listdir(self._img_paths[key])

        elif len(config["target"]) == 0:
            for key in self.sub_folders:
                self._targets += os.listdir(self._img_paths[key])

        if len(config["source"]) > 0:
            for key in config["source"]:
                self._sources += os.listdir(self._img_paths[key])

        elif len(config["source"]) == 0:
            for key in self.sub_folders:
                self._sources += os.listdir(self._img_paths[key])
        
        if len(config["segs"]) > 0:
            for key in config["segs"]:
                self._segs += os.listdir(self._seg_paths[key])

        if len(self._targets) == 0 or len(self._sources) == 0:
            raise FileNotFoundError(f"No data found: {len(self._targets)} targets, {len(self._sources)} sources")

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources, {len(self._segs)} segmentations")
        print(f"Using paired loader for {self._dataset_type}")

        super().train_val_split()

        self._subject_targets = {k: [img for img in v if img[6:8] in self.config["target"]] for k, v in self._subject_imgs.items()}
        self._subject_sources = {k: [img for img in v if img[6:8] in self.config["source"]] for k, v in self._subject_imgs.items()}
    
    @property
    def subject_imgs(self):
        """ Return list of images indexed by procedure """
        return {"targets": self._subject_targets, "sources": self._subject_sources}

    def img_pairer(self, source: str) -> dict:
        # TODO: return idx
        # Get potential target candidates matching source (where target and source specified)
        target_candidates = self._subject_targets[source[0:6]]
        # target_stem = target_candidates[np.random.randint(len(target_candidates))]
        target = f"{target_stem}.npy"

        return {"target": target, "source": source}

#-------------------------------------------------------------------------
""" Data loader for unpaired images """

class UnpairedLoader(BaseImgLoader):
    def __init__(self, config: dict, dataset_type: str):
        super().__init__(config, dataset_type)

        # Optional list of targets and sources e.g. ["AC", "VC"], ["HQ"]
        self._targets = []
        self._sources = []
        self._segs = []

        if len(config["target"]) > 0:
            for key in config["target"]:
                self._targets += [t for t in os.listdir(self._img_paths) if key in t]

        elif len(config["target"]) == 0:
            self._targets += os.listdir(self._img_paths)

        if len(config["source"]) > 0:
            for key in config["source"]:
                self._sources += [s for s in os.listdir(self._img_paths) if key in s]

        elif len(config["source"]) == 0:
            self._sources += os.listdir(self._img_paths)
        
        if len(config["segs"]) > 0:
            self._segs += os.listdir(self._seg_paths)

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources, {len(self._segs)} segmentations")
        print(f"Using unpaired loader for {self._dataset_type}")

        super().train_val_split()

        if len(self.config["target"]) > 0:
            self._subject_targets = {k: [img for img in v if img[6:8] in self.config["target"]] for k, v in self._subject_imgs.items()}
        else:
            self._subject_targets = None

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


#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":
    import yaml

    """ Routine for visually testing dataloader """

    test_config = yaml.load(open("syntheticcontrast_v02/utils/test_config.yml", 'r'), Loader=yaml.FullLoader)

    TestLoader = UnpairedLoader(config=test_config["data"], dataset_type="training")
    _, _ = TestLoader.set_normalisation()

    output_types = ["real_source", "real_target"]

    if len(test_config["data"]["segs"]) > 0:
        output_types += ["seg"]
    
    if test_config["data"]["times"] is not None:
        output_types += ["source_times", "target_times"]
    
    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types={k: "float32" for k in output_types})

    for data in train_ds.batch(2).take(16):
        source = TestLoader.un_normalise(data["real_source"])
        target = TestLoader.un_normalise(data["real_target"])

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["source_times"][0].numpy())

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["source_times"][1].numpy())

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["target_times"][0].numpy())

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, :, :, 0, 0].numpy(), cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(data["target_times"][1].numpy())

        if "seg" in data.keys():
            plt.subplot(3, 2, 5)
            plt.imshow(data["seg"][0, :, :, 0, 0].numpy())
            plt.axis("off")
            plt.subplot(3, 2, 6)
            plt.imshow(data["seg"][1, :, :, 0, 0].numpy())
            plt.axis("off")

        plt.show()

    data = TestLoader.example_images()

    source = TestLoader.un_normalise(data["real_source"])
    target = TestLoader.un_normalise(data["real_target"])
    
    fig, axs = plt.subplots(target.shape[0], 3)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        if "source_times" in data.keys():
            axs[i, 0].set_title(data["source_times"][i])

        axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")

        if "source_times" in data.keys():
            axs[i, 1].set_title(data["target_times"][i])

        if "seg" in data.keys():
            axs[i, 2].imshow(data["seg"][i, :, :, 11, 0])
            axs[i, 2].axis("off")
    
    plt.show()
