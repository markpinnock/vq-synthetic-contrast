import numpy as np
import tensorflow as tf

from vq_sce import RANDOM_SEED
from vq_sce.utils.augmentation.augmentation import StdAug

tf.random.set_seed(RANDOM_SEED)

PCT_CORRECT_THRESHOLD = 0.6
DEFAULT_IMG_SIZE = [4, 6, 8]
NUM_HOMOGENOUS_DIMS = 3

TEST_CE_DIMS = [
    [2, 32, 128, 128, 1],
    [4, 32, 128, 128, 1],
    [4, 64, 256, 256, 1]
]

TEST_HQ_DIMS = [
    [2, 12, 64, 64, 1],
    [4, 12, 64, 64, 1],
    [4, 12, 128, 128, 1]
]

TEST_LQ_DIMS = [
    [2, 3, 64, 64, 1],
    [4, 3, 64, 64, 1],
    [4, 3, 128, 128, 1]
]


#-------------------------------------------------------------------------

class TestAffine2D(tf.test.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.config = {
            "flip_prob": 1.0,
            "rotation": 45.0,
            "scale": [0.8, 1.6],
            "shear": 15.0,
            "translate": [0.25, 0.25]
        }
    
    def tearDown(self) -> None:
        super().tearDown()

    def create_test_img(self, img_dims: list[int]) -> tf.Tensor:
        img = np.zeros(img_dims)
        img[:, :, 0:img.shape[2] // 2, 0:img.shape[3] // 2, :] = 1
        img[:, :, -img.shape[2] // 2:, -img.shape[3] // 2:, :] = 1

        return tf.convert_to_tensor(img)

    def test_source_target_same(self) -> None:
        """ Test that source and targets are augmented identically """

        for CE_dim in TEST_CE_DIMS:
            with self.subTest():
                self.config["source_dims"] = CE_dim[1:-1]
                self.config["target_dims"] = CE_dim[1:-1]
                std_aug = StdAug(self.config)

                source = [
                    self.create_test_img(CE_dim),
                    self.create_test_img(CE_dim)
                ]
                target = [
                    self.create_test_img(CE_dim),
                    self.create_test_img(CE_dim)
                ]

                aug_source, aug_target = std_aug(source, target)

                for s, t in zip(aug_source, aug_target):
                    self.assertAllEqual(s, t)

    def test_mb_transforms_different(self) -> None:
        """ Test augmentations within minibatch are different """

        for CE_dim in TEST_CE_DIMS:
            with self.subTest():
                self.config["source_dims"] = CE_dim[1:-1]
                self.config["target_dims"] = CE_dim[1:-1]
                std_aug = StdAug(self.config)

                source = [self.create_test_img(CE_dim)]
                target = [self.create_test_img(CE_dim)]

                (aug_source,), (aug_target,) = std_aug(source, target)

                for i in range(1, CE_dim[0]):
                    self.assertNotAllEqual(
                        aug_source[i - 1, ...],
                        aug_source[i, ...]
                    )
                    self.assertNotAllEqual(
                        aug_target[i - 1, ...],
                        aug_target[i, ...]
                    )

    def test_rpt_transforms_different(self) -> None:
        """ Test sequential augmentations are different """

        self.config["source_dims"] = TEST_CE_DIMS[1][1:-1]
        self.config["target_dims"] = TEST_CE_DIMS[1][1:-1]
        std_aug = StdAug(self.config)

        source = [self.create_test_img(TEST_CE_DIMS[1])]
        target = [self.create_test_img(TEST_CE_DIMS[1])]

        (aug_source,), (aug_target,) = std_aug(source, target)
        (new_source,), (new_target,) = std_aug(source, target)

        self.assertNotAllEqual(aug_source, new_source)
        self.assertNotAllEqual(aug_target, new_target)

    def test_differing_source_target(self) -> None:
        """ Test different sized source and target augmented correctly """

        for LQ_dim, HQ_dim in zip(TEST_LQ_DIMS, TEST_HQ_DIMS):
            with self.subTest():
                self.config["source_dims"] = LQ_dim[1:-1]
                self.config["target_dims"] = HQ_dim[1:-1]
                std_aug = StdAug(self.config)

                source = [
                    self.create_test_img(LQ_dim),
                    self.create_test_img(LQ_dim)
                ]
                target = [
                    self.create_test_img(HQ_dim),
                    self.create_test_img(HQ_dim)
                ]

                aug_source, aug_target = std_aug(source, target)

                for s, t in zip(aug_source, aug_target):
                    self.assertAllEqual(s[:, 0, ...], t[:, 0, ...])
                    self.assertAllEqual(s[:, -1, ...], t[:, -1, ...])
