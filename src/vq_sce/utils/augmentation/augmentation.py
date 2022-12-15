import tensorflow as tf

from vq_sce.utils.augmentation.affine_transform import AffineTransform2D


#-------------------------------------------------------------------------
""" Standard augmentation performing flipping, rotating, scale and shear """

class StdAug(tf.keras.layers.Layer):

    def __init__(self, config, name="std_aug"):
        super().__init__(name=name)

        # If segmentations available, these can be stacked on the target for transforming
        if "segs" in config.keys():
            self.transform = AffineTransform2D(config["img_dims"] + [2])
        else:
            self.transform = AffineTransform2D(config["img_dims"] + [1])

        self.flip_probs = tf.math.log([[config["flip_prob"], 1 - config["flip_prob"]]])
        self.rot_angle = config["rotation"] / 180 * 3.14159265359
        self.scale_factor = config["scale"]
        self.shear_angle = config["shear"] / 180 * 3.14159265359
        self.x_shift = [-config["translate"][0], config["translate"][0]]
        self.y_shift = [-config["translate"][1], config["translate"][1]]

    def flip_matrix(self, mb_size: int):
        updates = tf.reshape(tf.cast(tf.random.categorical(logits=self.flip_probs, num_samples=mb_size * 2), "float32"), [mb_size * 2])
        updates = 2.0 * updates - 1.0
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])], axis=1)
        flip_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return flip_mat

    def rotation_matrix(self, mb_size: int):
        thetas = tf.random.uniform([mb_size], -self.rot_angle, self.rot_angle)
        rot_mat = tf.stack(
            [
                [tf.math.cos(thetas), -tf.math.sin(thetas)],
                [tf.math.sin(thetas), tf.math.cos(thetas)]
            ]
        )

        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat

    def scale_matrix(self, mb_size: int):
        updates = tf.repeat(tf.random.uniform([mb_size], * self.scale_factor), 2)
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])], axis=1)
        scale_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return scale_mat

    def shear_matrix(self, mb_size: int):
        mask = tf.cast(tf.random.categorical(logits=[[0.5, 0.5]], num_samples=mb_size), "float32")
        mask = tf.reshape(tf.transpose(tf.concat([mask, 1 - mask], axis=0), [1, 0]), [1, -1])
        updates = tf.repeat(tf.random.uniform([mb_size], -self.shear_angle, self.shear_angle), 2)
        updates = tf.reshape(updates * mask, [-1])
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 1], [1, 0]]), [mb_size, 1])], axis=1)
        shear_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])
        shear_mat += tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return shear_mat

    def translation_matrix(self, m, mb_size: int):
        xs = tf.random.uniform([mb_size], *self.x_shift)
        ys = tf.random.uniform([mb_size], *self.y_shift)
        xys = tf.stack([xs, ys], axis=0)
        xys = tf.transpose(xys, [1, 0])[:, :, tf.newaxis]
        m = tf.concat([m, xys], axis=2)

        return m

    def transformation(self, mb_size: int):
        trans_mat = tf.matmul(
            self.shear_matrix(mb_size), tf.matmul(
                self.flip_matrix(mb_size), tf.matmul(
                    self.rotation_matrix(mb_size), self.scale_matrix(mb_size))))

        trans_mat = self.translation_matrix(trans_mat, mb_size)

        return trans_mat
    
    def call(self, imgs, seg=None):
        l = len(imgs)
        imgs = tf.concat(imgs, axis=4)
        mb_size = imgs.shape[0]
        thetas = tf.reshape(self.transformation(mb_size), [mb_size, -1])

        if seg is not None:
            img_seg = tf.concat([imgs, seg], axis=4)
            img_seg = self.transform(im=img_seg, mb_size=mb_size, thetas=thetas)
            imgs = [img_seg[:, :, :, :, i][:, :, :, :, tf.newaxis] for i in range(l)]
            seg = img_seg[:, :, :, :, -1][:, :, :, :, tf.newaxis]

            return tuple(imgs), seg
        
        else:
            imgs = self.transform(im=imgs, mb_size=mb_size, thetas=thetas)
            imgs = [imgs[:, :, :, :, i][:, :, :, :, tf.newaxis] for i in range(l)]
            return tuple(imgs)


#-------------------------------------------------------------------------
""" Short routine for visually testing augmentations """

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from vq_sce.utils.dataloaders.contrast_dataloader import ContrastDataloader
    from vq_sce.utils.dataloaders.super_res_dataloader import SuperResDataloader

    test_config = yaml.load(
        open("src/vq_sce/utils/test_config.yml", 'r'),
        Loader=yaml.FullLoader
    )

    contrast_loader = ContrastDataloader(
        test_config["data"],
        dataset_type="training"
    )
    super_res_loader = SuperResDataloader(
        test_config["data"],
        dataset_type="training"
    )

    output_types = ["source", "target"]
    contrast_ds = tf.data.Dataset.from_generator(
        contrast_loader.data_generator,
        output_types={k: "float32" for k in output_types}
    )
    super_res_ds = tf.data.Dataset.from_generator(
        super_res_loader.data_generator,
        output_types={k: "float32" for k in output_types}
    )

    aug_config = test_config["augmentation"]
    TestAug = StdAug(aug_config)

    for data in contrast_ds.batch(2).take(2):
        pred = np.zeros_like(data["source"].numpy())
        pred[:, 0:pred.shape[1] // 2, 0:pred.shape[1] // 2, :, :] = 1
        pred[:, pred.shape[1] // 2:, pred.shape[1] // 2:, :, :] = 1
        inv_pred = 1 - pred

        source, target, pred, inv_pred = TestAug(
            [data["source"],
            data["target"],
            pred,
            inv_pred]
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

    for data in super_res_ds.batch(2).take(2):
        pred = np.zeros_like(data["source"].numpy())
        pred[:, 0:pred.shape[1] // 2, 0:pred.shape[1] // 2, :, :] = 1
        pred[:, pred.shape[1] // 2:, pred.shape[1] // 2:, :, :] = 1
        inv_pred = 1 - pred

        source, target, pred, inv_pred = TestAug(
            [data["source"],
            data["target"],
            pred,
            inv_pred]
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