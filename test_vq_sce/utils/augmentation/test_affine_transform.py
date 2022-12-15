import numpy as np
import tensorflow as tf
from typing import Any, Callable, TypeVar

from vq_sce import RANDOM_SEED
from vq_sce.utils.augmentation.affine_transform import AffineTransform2D

PCT_CORRECT_THRESHOLD = 0.6
DEFAULT_IMG_SIZE = [4, 6, 8]
NUM_HOMOGENOUS_DIMS = 3
TEST_IMG_DIMS_2D = [
    [2, 64, 64, 1],
    [4, 64, 64, 1],
    [4, 128, 128, 1],
    [4, 128, 128, 3]
]
TEST_IMG_DIMS_3D = [
    [2, 32, 64, 64, 1],
    [4, 32, 64, 64, 1],
    [4, 64, 128, 128, 1],
    [4, 64, 128, 128, 3]
]

Self = TypeVar("Self")


#-------------------------------------------------------------------------
""" Parameterize decorator as pytest does not work with tf.test.TestCase """

def parametrize(args: list[Any]):
    def decorator(func: Callable):
        def wrapper(s: Self, *args: list[Any]):
            for arg in args:
                func(s, *arg)
        return wrapper
    return decorator


#-------------------------------------------------------------------------

class TestAffine2D(tf.test.TestCase):

    def setUp(self) -> None:
        super().setUp()
    
    def tearDown(self) -> None:
        super().tearDown()

    def test_coord_gen(self) -> None:
        """ Test flat image coordinates """

        # 2D case
        img_dims = [(2, 3), (5, 6), (512, 512)]
        for dim in img_dims:
            with self.subTest():
                affine = AffineTransform2D(img_dims=dim)
                exp_flat_coords_shape = (NUM_HOMOGENOUS_DIMS, np.prod(dim))
                self.assertAllEqual(affine.flat_coords.shape, exp_flat_coords_shape)

        # 3D case
        img_dims = [(1, 2, 3), (4, 5, 6), (64, 512, 512)]
        for dim in img_dims:
            with self.subTest():
                affine = AffineTransform2D(img_dims=dim)
                exp_flat_coords_shape = (NUM_HOMOGENOUS_DIMS, np.prod(dim[1:]))
                self.assertAllEqual(affine.flat_coords.shape, exp_flat_coords_shape)

    def test_transform_coords(self) -> None:
        """ Test transforming flat image coordinates """

        thetas = [
            np.array([[1, 0, 0], [0, 1, 0]]).astype("float32"),
            np.array([[1, 0, -1], [0, 1, 2]]).astype("float32"),
            np.array([[1, 5, -1], [3, 1, 2]]).astype("float32")
        ]
        affine = AffineTransform2D(DEFAULT_IMG_SIZE)

        for theta in thetas:
            with self.subTest():
                flat_coords = affine.flat_coords.numpy()
                new_coords = theta @ flat_coords

                affine.transform_coords(1, tf.constant(theta.reshape([1, -1])))
                Y, X = affine.mesh_coords

                self.assertAllClose(X, new_coords[0, :])
                self.assertAllClose(Y, new_coords[1, :])

    def test_transform_coords_mb(self) -> None:
        """ Test transforming minibatch of flat image coordinates """

        mb_sizes = [2, 4]
        affine = AffineTransform2D(DEFAULT_IMG_SIZE)

        for mb_size in mb_sizes:
            with self.subTest():
                np.random.seed(RANDOM_SEED)
                thetas = np.random.randint(-5, 5, size=[mb_size, 2, 3]).astype("float32")
                np.random.seed()

                flat_coords = affine.flat_coords.numpy()
                new_coords = []

                for i in range(mb_size):
                    new_coords.append(thetas[i, :, :] @ flat_coords)
                
                affine.transform_coords(mb_size, tf.constant(thetas.reshape(mb_size, -1)))

                Y, X = affine.mesh_coords
                ground_truth_X = np.hstack([c[0, :] for c in new_coords])
                ground_truth_Y = np.hstack([c[1, :] for c in new_coords])
                self.assertAllEqual(X, ground_truth_X)
                self.assertAllEqual(Y, ground_truth_Y)

    def setup_flipping_img(self, img_dims, horizontal, vertical) -> None:
        """ Create fixtures for flipping tests """

        if img_dims[0] == 2:
            thetas = [
                np.array([-1, 0, 0, 0, 1, 0]),
                np.array([1, 0, 0, 0, -1, 0])
            ]

            mb = tf.convert_to_tensor(
                np.stack([horizontal, vertical], axis=0).astype("float32")
            )
            gt_mb = 1 - mb

        else:
            thetas = [
                np.array([-1, 0, 0, 0, 1, 0]),
                np.array([-1, 0, 0, 0, 1, 0]),
                np.array([1, 0, 0, 0, -1, 0]),
                np.array([1, 0, 0, 0, -1, 0])
            ]

            mb = tf.convert_to_tensor(
                np.stack(
                    [
                        horizontal, vertical, horizontal, vertical
                    ], axis=0
                ).astype("float32")
            )

            gt_mb = tf.convert_to_tensor(
                np.stack(
                    [
                        1 - horizontal, vertical, horizontal, 1 - vertical
                    ], axis=0
                ).astype("float32")
            )

        thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))
        return thetas, mb, gt_mb

    def test_flipping_2D(self) -> None:
        """ Test horizontal/vertical flipping for 2D images """

        for img_dims in TEST_IMG_DIMS_2D:
            with self.subTest():
                # Create test images divided vertically and horizontally
                vertical = np.zeros(img_dims[1:])
                vertical[0:img_dims[1] // 2, :, :] = 1
                horizontal = np.zeros(img_dims[1:])
                horizontal[:, 0:img_dims[2] // 2, :] = 1

                thetas, mb, gt_mb = self.setup_flipping_img(img_dims, horizontal, vertical)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

    def test_flipping_3D(self) -> None:
        """ Test 2D horizontal/vertical flipping for 3D images """

        for img_dims in TEST_IMG_DIMS_3D:
            with self.subTest():
                # Create test images divided vertically and horizontally
                vertical = np.zeros(img_dims[1:])
                vertical[:, 0:img_dims[1] // 2, :, :] = 1
                horizontal = np.zeros(img_dims[1:])
                horizontal[:, :, 0:img_dims[2] // 2, :] = 1

                thetas, mb, gt_mb = self.setup_flipping_img(img_dims, horizontal, vertical)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

    def setup_rotation_img(self, img_dims, base_img) -> None:
        """ Create fixtures for rotation tests """

        if img_dims[0] == 2:
            thetas = [
                np.array([0, 1, 0, -1, 0, 0]),
                np.array([-1, 0, 0, 0, -1, 0])
            ]

            mb = tf.convert_to_tensor(
                np.stack([base_img, base_img], axis=0).astype("float32")
            )
            gt_mb = tf.convert_to_tensor(
                np.stack([1 - base_img, base_img], axis=0).astype("float32")
            )

        else:
            thetas = [
                np.array([1, 0, 0, 0, 1, 0]),
                np.array([0, 1, 0, -1, 0, 0]),
                np.array([-1, 0, 0, 0, -1, 0]),
                np.array([0, -1, 0, 1, 0, 0])
            ]

            mb = tf.convert_to_tensor(
                np.stack([
                        base_img, base_img, base_img, base_img
                    ], axis=0
                ).astype("float32")
            )

            gt_mb = tf.convert_to_tensor(
                np.stack(
                    [
                        base_img, 1 - base_img, base_img, 1 - base_img
                    ], axis=0
                ).astype("float32")
            )

        thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))
        return thetas, mb, gt_mb

    def test_rotation_2D(self) -> None:
        """ Test rotation for 2D images """

        for img_dims in TEST_IMG_DIMS_2D:
            with self.subTest():
                # Create test images divided into quadrants
                base_img = np.zeros(img_dims[1:])
                base_img[0:img_dims[1] // 2, 0:img_dims[1] // 2, :] = 1
                base_img[-img_dims[1] // 2:, -img_dims[1] // 2:, :] = 1

                thetas, mb, gt_mb = self.setup_rotation_img(img_dims, base_img)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

    def test_rotation_3D(self) -> None:
        """ Test 2D rotation for 3D images """

        for img_dims in TEST_IMG_DIMS_3D:
            with self.subTest():
                # Create test images divided into quadrants
                base_img = np.zeros(img_dims[1:])
                base_img[:, 0:img_dims[1] // 2, 0:img_dims[1] // 2, :] = 1
                base_img[:, -img_dims[1] // 2:, -img_dims[1] // 2:, :] = 1

                thetas, mb, gt_mb = self.setup_rotation_img(img_dims, base_img)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

    def setup_scaling_img(self, img_dims, img_small) -> None:
        """ Create fixtures for scaling tests """

        img_large = np.ones(img_dims[1:])

        if img_dims[0] == 2:
            thetas = [
                np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
                np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
            ]

            mb = tf.convert_to_tensor(
                np.stack([img_large, img_small], axis=0).astype("float32")
            )
            gt_mb = tf.convert_to_tensor(
                np.stack([img_small, img_large], axis=0).astype("float32")
            )

        else:
            thetas = [
                np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]),
                np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
                np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
            ]

            mb = tf.convert_to_tensor(
                np.stack([
                        img_large, img_small, img_large, img_small
                    ], axis=0
                ).astype("float32")
            )

            gt_mb = tf.convert_to_tensor(
                np.stack(
                    [
                        img_small, img_large, img_small, img_large
                    ], axis=0
                ).astype("float32")
            )

        # if img_dims[1] == 1:
        #     mb = tf.reshape(mb, [N, H, W, C])
        #     gt_mb = tf.reshape(gt_mb, [N, H, W, C])

        thetas = tf.convert_to_tensor(np.stack(thetas, axis=0).astype("float32"))
        return thetas, mb, gt_mb

    def test_scaling_2D(self) -> None:
        """ Test scaling for 2D images """

        for img_dims in TEST_IMG_DIMS_3D:
            with self.subTest():
                N = img_dims[0]
                H = img_dims[2]
                W = img_dims[3]
                C = img_dims[4]
                mid_h = H // 2
                mid_w = W // 2
                img_small = np.zeros(img_dims[1:])

                img_small[
                    :, mid_h - H // 4:mid_h + H // 4,
                    mid_w - W // 4:mid_w + W // 4, :
                ] = 1

                thetas, mb, gt_mb = self.setup_scaling_img(img_dims, img_small)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

    def test_scaling_3D(self) -> None:
        """ Test 2D scaling for 3D images """

        for img_dims in TEST_IMG_DIMS_3D:
            with self.subTest():
                N = img_dims[0]
                H = img_dims[2]
                W = img_dims[3]
                C = img_dims[4]
                mid_h = H // 2
                mid_w = W // 2
                img_small = np.zeros(img_dims[1:])

                img_small[
                    :, mid_h - H // 4:mid_h + H // 4,
                    mid_w - W // 4:mid_w + W // 4, :
                ] = 1

                thetas, mb, gt_mb = self.setup_scaling_img(img_dims, img_small)
                affine = AffineTransform2D(img_dims[1:-1])
                new_mb = affine(mb, thetas)

                # Because of rounding errors at e.g. boundaries, we can't compare directly
                pct_correct_voxels = np.sum(new_mb.numpy() == gt_mb.numpy()) / np.prod(new_mb.shape)
                assert pct_correct_voxels > PCT_CORRECT_THRESHOLD

if __name__ == "__main__":
    tf.test.main()
