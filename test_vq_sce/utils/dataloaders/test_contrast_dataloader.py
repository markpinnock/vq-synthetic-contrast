import json
import numpy as np
from pathlib import Path
import pytest
import shutil

from vq_sce import HU_MIN, HU_MAX, RANDOM_SEED
from vq_sce.utils.dataloaders.build_dataloader import ContrastDataloader

NUM_IMAGES = 15
IMG_SIZE = [4, 8, 8]
TEST_PATH = Path(__file__).parent / "data"
BASE_CONFIG = {
    "data_path": TEST_PATH,
    "down_sample": 1,
    "patch_size": [4, 4, 4],
    "num_examples": 4,
    "cv_folds": 3,
    "fold": 2
}

#-------------------------------------------------------------------------

@pytest.fixture(scope="function", autouse=True)
def _setup_dataset() -> None:
    target_path = TEST_PATH / "CE"
    source_path = TEST_PATH / "HQ"
    target_path.mkdir(parents=True, exist_ok=True)
    source_path.mkdir(parents=True, exist_ok=True)

    source_coords = {}
    np.random.seed(RANDOM_SEED)

    for i in range(NUM_IMAGES):
        img = np.random.randint(HU_MIN, HU_MAX, IMG_SIZE)
        img[0, 0, 0] = HU_MIN
        img[-1, -1, -1] = HU_MAX
        np.save(target_path / f"{i}.npy", img)
        np.save(source_path / f"{i}.npy", img)
        source_coords[f"{i}"] = {f"{i}": [0, IMG_SIZE[0]]}

    with open(TEST_PATH / "source_coords.json", 'w') as fp:
        json.dump(source_coords, fp)

    np.random.seed()

    yield

    shutil.rmtree(TEST_PATH)


#-------------------------------------------------------------------------
""" Test if exception raised if wrong data path given """

def test_wrong_path() -> None:
    config = dict(BASE_CONFIG)
    config["data_path"] = "."

    with pytest.raises(FileNotFoundError):
        _ = ContrastDataloader(config=config, dataset_type="training")


#-------------------------------------------------------------------------
""" Test if exception raised if incorrect subset given """

def test_wrong_subset() -> None:
    with pytest.raises(ValueError):
        _ = ContrastDataloader(config=BASE_CONFIG, dataset_type="test")


#-------------------------------------------------------------------------
""" Test if exception raised if incorrect fold params given """

@pytest.mark.parametrize("fold,cv_folds", [(5, 5), (0, 0)])
def test_wrong_fold_params(fold: int, cv_folds: int) -> None:
    config = dict(BASE_CONFIG)
    config["cv_folds"] = cv_folds
    config["fold"] = fold

    with pytest.raises(AssertionError):
        _ = ContrastDataloader(config=config, dataset_type="training")


#-------------------------------------------------------------------------
""" Test train/validation split size """

@pytest.mark.parametrize(
    "subset,fold,cv_folds,exp_num_images",
    [
        ("training", 0, 3, 10),
        ("training", 1, 3, 10),
        ("training", 2, 3, 10),
        ("training", 0, 5, 12),
        ("validation", 0, 3, 5),
        ("validation", 1, 3, 5),
        ("validation", 2, 3, 5),
        ("validation", 0, 5, 3),
    ]
)
def test_train_val_split_size(
    subset: str,
    fold: int,
    cv_folds: int,
    exp_num_images: int
) -> None:

    config = dict(BASE_CONFIG)
    config |= {"fold": fold, "cv_folds": cv_folds}

    dataloader = ContrastDataloader(config=config, dataset_type=subset)
    assert len(dataloader.data["source"]) == exp_num_images


#-------------------------------------------------------------------------
""" Test train/validation splits are mutually exclusive """

def test_train_val_exclusive() -> None:

    train_dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="training"
    )
    valid_dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="validation"
    )

    train_ids = list(train_dataloader.data["source"].keys())
    valid_ids = list(valid_dataloader.data["source"].keys())

    for id_ in valid_ids:
        assert id_ not in train_ids


#-------------------------------------------------------------------------
""" Test train/validation splits are non-random """

@pytest.mark.parametrize("subset", ["training", "validation"])
def test_train_val_non_random(subset: str) -> None:

    dataloader1 = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)
    dataloader2 = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)

    ids1 = list(dataloader1.data["source"].keys())
    ids2 = list(dataloader2.data["source"].keys())
    assert len(ids1) == len(ids2)

    for id_ in ids1:
        assert id_ in ids2


#-------------------------------------------------------------------------
""" Test train/validation splits are correctly paired """

def test_train_val_pairing() -> None:

    dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="training"
    )

    source_ids = list(dataloader.data["source"].keys())
    target_ids = list(dataloader.data["target"].keys())

    for source_id, target_id in zip(source_ids, target_ids):
        assert source_id == target_id


#-------------------------------------------------------------------------
""" Test number of example images """

@pytest.mark.parametrize("num_examples", [2, 4])
def test_example_image_number(num_examples: int) -> None:

    config = dict(BASE_CONFIG)
    config |= {"num_examples": num_examples}

    dataloader = ContrastDataloader(config=config, dataset_type="training")
    assert dataloader.example_images["source"].shape[0] == num_examples


#-------------------------------------------------------------------------
""" Test train/validation example images are mutually exclusive """

def test_example_image_exclusive() -> None:

    config = dict(BASE_CONFIG)
    config |= {"patch_size": [2, 4, 4]}

    train_dataloader = ContrastDataloader(
        config=config,
        dataset_type="training"
    )
    valid_dataloader = ContrastDataloader(
        config=config,
        dataset_type="validation"
    )

    train_imgs = train_dataloader.example_images["source"]
    valid_imgs = valid_dataloader.example_images["source"]

    assert not np.equal(train_imgs, valid_imgs).all()


#-------------------------------------------------------------------------
""" Test train/validation example images are non-random """

@pytest.mark.parametrize("subset", ["training", "validation"])
def test_example_image_non_random(subset: str) -> None:

    dataloader1 = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)
    dataloader2 = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)

    imgs1 = dataloader1.example_images["source"]
    imgs2 = dataloader2.example_images["source"]

    assert np.equal(imgs1, imgs2).all()


#-------------------------------------------------------------------------
""" Test example images are properly paired """

def test_example_image_pairing():
    dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="training"
    )

    source_imgs = dataloader.example_images["source"]
    target_imgs = dataloader.example_images["target"]

    assert np.equal(source_imgs, target_imgs).all()


#-------------------------------------------------------------------------
""" Test images are correctly sized """

@pytest.mark.parametrize(
    "down_sample,patch_size",
    [
        (1, [1, 4, 4]),
        (1, [2, 4, 4]),
        (1, [4, 4, 4]),
        (2, [2, 4, 4]),
        (4, [2, 4, 4])
    ]
)
def test_generator_img_size(down_sample: int, patch_size: list[int]):

    config = dict(BASE_CONFIG)
    config |= {"down_sample": down_sample, "patch_size": patch_size}
    exp_img_size = (
        patch_size[0],
        IMG_SIZE[1] // down_sample,
        IMG_SIZE[2] // down_sample,
        1
    )

    dataloader = ContrastDataloader(config=config, dataset_type="training")

    for data in dataloader.data_generator():
        source = data["source"]
        target = data["target"]
        assert source.shape == exp_img_size
        assert target.shape == exp_img_size


#-------------------------------------------------------------------------
""" Test images are shuffled each training epoch """

def test_train_generator_random():
    dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="training"
    )

    imgs1, imgs2 = [], []

    for data in dataloader.data_generator():
        imgs1.append(data["source"])

    for data in dataloader.data_generator():
        imgs2.append(data["source"])

    imgs1 = np.stack(imgs1)
    imgs2 = np.stack(imgs2)

    assert not np.equal(imgs1, imgs2).all()


#-------------------------------------------------------------------------
""" Test images are not shuffled each validation epoch """

def test_valid_generator_nonrandom():
    dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="validation"
    )

    imgs1, imgs2 = [], []

    for data in dataloader.data_generator():
        imgs1.append(data["source"])

    for data in dataloader.data_generator():
        imgs2.append(data["source"])

    imgs1 = np.stack(imgs1)
    imgs2 = np.stack(imgs2)

    assert np.equal(imgs1, imgs2).all()


#-------------------------------------------------------------------------
""" Test images voxel values are in range [-1, 1] """

@pytest.mark.parametrize("subset", ["training", "validation"])
def test_generator_voxel_values(subset):

    dataloader = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)

    for data in dataloader.data_generator():
        assert np.max(data["source"]) == 1.0
        assert np.min(data["source"]) == -1.0
        assert np.max(data["target"]) == 1.0
        assert np.min(data["target"]) == -1.0


#-------------------------------------------------------------------------
""" Test un-normalised voxel values are between HU_MIN and HU_MAX """

def test_un_normalise():

    dataloader = ContrastDataloader(
        config=BASE_CONFIG,
        dataset_type="training"
    )

    for data in dataloader.data_generator():
        source = dataloader.un_normalise(data["source"])
        target = dataloader.un_normalise(data["target"])

        assert np.max(source) == HU_MAX
        assert np.min(source) == HU_MIN
        assert np.max(target) == HU_MAX
        assert np.min(target) == HU_MIN


#-------------------------------------------------------------------------
""" Test images paired correctly """

@pytest.mark.parametrize("subset", ["training", "validation"])
def test_generator_pairing(subset):

    dataloader = ContrastDataloader(config=BASE_CONFIG, dataset_type=subset)

    for data in dataloader.data_generator():
        assert np.equal(data["source"], data["target"]).all()
