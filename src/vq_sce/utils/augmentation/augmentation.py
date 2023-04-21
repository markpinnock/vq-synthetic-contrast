from typing import Any

import numpy as np
import tensorflow as tf

from vq_sce.utils.augmentation.affine_transform import AffineTransform2D

DEG_TO_RAD = 180 * np.pi


# -------------------------------------------------------------------------


class StdAug(tf.keras.layers.Layer):
    """Standard augmentation performing flipping, rotating, scale and shear."""

    def __init__(self, config: dict[str, Any], name: str = "std_aug") -> None:
        super().__init__(name=name)

        self.source_affine_transform = AffineTransform2D(config["source_dims"])
        self.target_affine_transform = AffineTransform2D(config["target_dims"])
        self._matrix_indices = None

        self._flip_probs = tf.math.log([[config["flip_prob"], 1 - config["flip_prob"]]])
        self._rot_angle = config["rotation"] / DEG_TO_RAD
        self._scale_factor = config["scale"]
        self._shear_angle = config["shear"] / DEG_TO_RAD
        self._x_shift = [-config["translate"][0], config["translate"][0]]
        self._y_shift = [-config["translate"][1], config["translate"][1]]

    def _create_matrix_indices(self, mb_size: int) -> None:
        """Create indices to update transformation matrix elements."""
        mb_indices = tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis]
        element_indices = tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])
        self._matrix_indices = tf.concat([mb_indices, element_indices], axis=1)

    def _flip_matrix(self, mb_size: int) -> tf.Tensor:
        """Create flip matrix."""
        updates = tf.random.categorical(
            logits=self._flip_probs,
            num_samples=mb_size * 2,
        )
        updates = tf.squeeze(tf.cast(updates, "float32"))
        updates = 2.0 * updates - 1.0

        # Create flip matrix
        flip_mat = tf.scatter_nd(self._matrix_indices, updates, [mb_size, 2, 2])

        return flip_mat

    def _rotation_matrix(self, mb_size: int) -> tf.Tensor:
        """Create rotation matrix."""
        thetas = tf.random.uniform([mb_size], -self._rot_angle, self._rot_angle)
        rot_mat = tf.stack(
            [
                [tf.math.cos(thetas), -tf.math.sin(thetas)],
                [tf.math.sin(thetas), tf.math.cos(thetas)],
            ],
        )

        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat

    def _scale_matrix(self, mb_size: int) -> tf.Tensor:
        """Create scaling matrix."""
        updates = tf.repeat(tf.random.uniform([mb_size], *self._scale_factor), 2)
        scale_mat = tf.scatter_nd(self._matrix_indices, updates, [mb_size, 2, 2])

        return scale_mat

    def _shear_matrix(self, mb_size: int) -> tf.Tensor:
        """Create shear matrix."""
        mask = tf.random.categorical(logits=[[0.0, 0.0]], num_samples=mb_size)
        mask = tf.cast(mask, "float32")
        mask = tf.transpose(tf.concat([mask, 1 - mask], axis=0), [1, 0])
        mask = tf.reshape(mask, [1, -1])

        updates = tf.random.uniform([mb_size], -self._shear_angle, self._shear_angle)
        updates = tf.repeat(updates, 2)
        updates = tf.reshape(updates * mask, [-1])

        # Can't use self._matrix_indices here, as not diagonal elements
        mb_indices = tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis]
        element_indices = tf.tile(tf.constant([[0, 1], [1, 0]]), [mb_size, 1])
        indices = tf.concat([mb_indices, element_indices], axis=1)

        shear_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])
        shear_mat += tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return shear_mat

    def _translation_matrix(self, matrix: tf.Tensor, mb_size: int) -> tf.Tensor:
        """Create translation matrix."""
        xs = tf.random.uniform([mb_size], *self._x_shift)
        ys = tf.random.uniform([mb_size], *self._y_shift)

        xys = tf.stack([xs, ys], axis=0)
        xys = tf.transpose(xys, [1, 0])[:, :, tf.newaxis]
        matrix = tf.concat([matrix, xys], axis=2)

        return matrix

    def _transformation(self, mb_size: int) -> tf.Tensor:
        trans_mat = tf.matmul(
            self._rotation_matrix(mb_size),
            self._scale_matrix(mb_size),
        )
        trans_mat = tf.matmul(self._flip_matrix(mb_size), trans_mat)
        trans_mat = tf.matmul(self._shear_matrix(mb_size), trans_mat)
        trans_mat = self._translation_matrix(trans_mat, mb_size)

        return trans_mat

    def call(
        self,
        source: list[tf.Tensor],
        target: list[tf.Tensor],
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        aug_source: list[tf.Tensor | None] = []
        aug_target: list[tf.Tensor | None] = []

        mb_size = tf.shape(source[0])[0]
        self._create_matrix_indices(mb_size)
        thetas = tf.reshape(self._transformation(mb_size), [mb_size, -1])

        for s in source:
            aug_source.append(self.source_affine_transform(s, thetas))

        for t in target:
            aug_target.append(self.target_affine_transform(t, thetas))

        return aug_source, aug_target


# -------------------------------------------------------------------------


if __name__ == "__main__":
    """Short routine for visually testing augmentations."""

    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from vq_sce.utils.dataloaders.contrast_dataloader import ContrastDataloader
    from vq_sce.utils.dataloaders.super_res_dataloader import SuperResDataloader

    test_config = yaml.load(
        open("src/vq_sce/utils/test_config.yml"),
        Loader=yaml.FullLoader,
    )

    contrast_loader = ContrastDataloader(test_config["data"], dataset_type="training")
    super_res_loader = SuperResDataloader(test_config["data"], dataset_type="training")

    output_types = ["source", "target"]
    contrast_ds = tf.data.Dataset.from_generator(
        contrast_loader.data_generator,
        output_types={k: "float32" for k in output_types},
    )
    super_res_ds = tf.data.Dataset.from_generator(
        super_res_loader.data_generator,
        output_types={k: "float32" for k in output_types},
    )

    aug_config = test_config["augmentation"]
    aug_config["source_dims"] = [12, 512, 512]
    aug_config["target_dims"] = [12, 512, 512]
    TestAug = StdAug(aug_config)

    for data in contrast_ds.batch(2).take(2):
        pred = np.zeros_like(data["source"].numpy())
        pred[:, :, 0 : pred.shape[2] // 2, 0 : pred.shape[3] // 2, :] = 1
        pred[:, :, -pred.shape[2] // 2 :, -pred.shape[3] // 2 :, :] = 1
        inv_pred = 1 - pred

        (source, pred), (target, inv_pred) = TestAug(
            [data["source"], pred],
            [data["target"], inv_pred],
        )

        plt.subplot(2, 6, 1)
        plt.imshow(data["source"][0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 7)
        plt.imshow(data["source"][1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 2)
        plt.imshow(source[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 8)
        plt.imshow(source[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 3)
        plt.imshow(target[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 9)
        plt.imshow(target[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 4)
        plt.imshow(pred[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 10)
        plt.imshow(pred[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 5)
        plt.imshow(inv_pred[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 11)
        plt.imshow(inv_pred[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.show()

    aug_config["source_dims"] = [3, 512, 512]
    aug_config["target_dims"] = [12, 512, 512]
    TestAug = StdAug(aug_config)

    for data in super_res_ds.batch(2).take(2):
        pred = np.zeros_like(data["target"].numpy())
        pred[:, :, 0 : pred.shape[2] // 2, 0 : pred.shape[3] // 2, :] = 1
        pred[:, :, -pred.shape[2] // 2 :, -pred.shape[3] // 2 :, :] = 1
        inv_pred = 1 - pred

        (source,), (target, pred, inv_pred) = TestAug(
            [data["source"]],
            [data["target"], pred, inv_pred],
        )

        plt.subplot(2, 6, 1)
        plt.imshow(data["source"][0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 7)
        plt.imshow(data["source"][1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 2)
        plt.imshow(source[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 8)
        plt.imshow(source[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 3)
        plt.imshow(target[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 9)
        plt.imshow(target[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 4)
        plt.imshow(pred[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 10)
        plt.imshow(pred[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 5)
        plt.imshow(inv_pred[0, 0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 11)
        plt.imshow(inv_pred[1, 0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.show()
