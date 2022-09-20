import tensorflow as tf

from vec_quant_sCE.utils.affinetransformation import AffineTransform2D


#-------------------------------------------------------------------------
""" Standard augmentation performing flipping, rotating, scale and shear """

class StdAug(tf.keras.layers.Layer):

    def __init__(self, config, name="std_aug"):
        super().__init__(name=name)

        # If segmentations available, these can be stacked on the target for transforming
        if len(config["data"]["segs"]) > 0:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [2])
        else:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [1])

        self.flip_probs = tf.math.log([[config["augmentation"]["flip_prob"], 1 - config["augmentation"]["flip_prob"]]])
        self.rot_angle = config["augmentation"]["rotation"] / 180 * 3.14159265359
        self.scale_factor = config["augmentation"]["scale"]
        self.shear_angle = config["augmentation"]["shear"] / 180 * 3.14159265359
        self.x_shift = [-config["augmentation"]["translate"][0], config["augmentation"]["translate"][0]]
        self.y_shift = [-config["augmentation"]["translate"][1], config["augmentation"]["translate"][1]]

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
            return tuple(imgs), None


#-------------------------------------------------------------------------
""" Short routine for visually testing augmentations """

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from vec_quant_sCE.utils.dataloader import ImgLoader

    test_config = yaml.load(open("vec_quant_sCE/utils/test_config.yml", 'r'), Loader=yaml.FullLoader)

    FILE_PATH = "D:/ProjectImages/SyntheticContrast"
    TestLoader = ImgLoader(test_config["data"], dataset_type="training")
    TestAug = StdAug(test_config)

    output_types = ["source", "target"]

    if len(test_config["data"]["segs"]) > 0:
        output_types += ["seg"]

    if test_config["data"]["times"] is not None:
        output_types += ["times"]

    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types={k: "float32" for k in output_types})

    for data in train_ds.batch(4):
        pred = np.zeros_like(data["source"].numpy())
        pred[:, 0:pred.shape[1] // 2, 0:pred.shape[1] // 2, :, :] = 1
        pred[:, pred.shape[1] // 2:, pred.shape[1] // 2:, :, :] = 1
        inv_pred = 1 - pred

        (source, target, pred, inv_pred), seg = TestAug([data["source"], data["target"], pred, inv_pred], seg=data["seg"])

        plt.subplot(2, 6, 1)
        plt.imshow(data["source"][0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 7)
        plt.imshow(data["source"][1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 2)
        plt.imshow(source[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 8)
        plt.imshow(source[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        
        plt.subplot(2, 6, 3)
        plt.imshow(target[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 9)
        plt.imshow(target[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 4)
        plt.imshow(pred[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 10)
        plt.imshow(pred[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(2, 6, 5)
        plt.imshow(inv_pred[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, 11)
        plt.imshow(inv_pred[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        if "seg" in data.keys():
            plt.subplot(2, 6, 6)
            plt.imshow(seg[0, :, :, 0, 0])
            plt.axis("off")
            plt.subplot(2, 6, 12)
            plt.imshow(seg[1, :, :, 0, 0])
            plt.axis("off")

        plt.show()
