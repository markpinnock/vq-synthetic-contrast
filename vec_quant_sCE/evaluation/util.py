import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import random
import tensorflow as tf


#-------------------------------------------------------------------------
""" Dataloader for segmentation-based evaluation - loads images .nrrd
    image volumes into RAM as 2D images """

def load_data(split, img_res, img_path, seg_path, img_type, ignore):
    HU_min = -500
    HU_max = 2500
    unique_ids = []
    train_test_split = 10

    if isinstance(img_type, float):
        time_str = str(img_type).replace('.', '_')

        for img_id in os.listdir(f"{img_path}/{time_str}"):
            if img_id[0:4] not in unique_ids and img_id[0:6] != ignore:
                unique_ids.append(img_id[0:4])

    else:
        for img_id in os.listdir(img_path):
            if img_id[0:4] not in unique_ids and img_id[0:6] != ignore:
                unique_ids.append(img_id[0:4])

    np.random.seed(5)
    np.random.shuffle(unique_ids)

    if split == "train":
        fold_ids = unique_ids[0:train_test_split]
    elif split == "test":
        fold_ids = unique_ids[train_test_split:]
    else:
        raise ValueError

    if isinstance(img_type, float):
        img_list = [img for img in os.listdir(f"{img_path}/{time_str}") if img[0:4] in fold_ids]
        seg_list = []

        for img in img_list:
            seg_list.append(f"{img[0:6]}HQ{img[8:11]}-label.nrrd")

    else:
        img_list = [img for img in os.listdir(img_path) if img_type in img and img[0:4] in fold_ids]
        seg_list = []

        for img in img_list:
            if img_type in ["AC", "VC"] and isinstance(img_type, str):
                seg_list.append(f"{img.split('.')[0]}-label.nrrd")
            else:
                seg_list.append(f"{img[0:6]}HQ{img[8:11]}-label.nrrd")

    imgs = []
    segs = []

    for img, seg in zip(img_list, seg_list):
        if isinstance(img_type, float):
            img_arr = nrrd.read(f"{img_path}/{time_str}/{img}")[0].astype("float32")

        else:
            img_arr = nrrd.read(f"{img_path}/{img}")[0].astype("float32")

        seg_arr = nrrd.read(f"{seg_path}/{seg}")[0].astype("float32")

        img_dims = img_arr.shape
        img_arr[img_arr < HU_min] = HU_min
        img_arr[img_arr > HU_max] = HU_max
        img_arr = (img_arr - HU_min) / (HU_max - HU_min)

        idx = np.argwhere(seg_arr == 1)
        x = (np.unique(idx[:, 0])[0], np.unique(idx[:, 0])[-1])
        y = (np.unique(idx[:, 1])[0], np.unique(idx[:, 1])[-1])
        z = (np.unique(idx[:, 2])[0], np.unique(idx[:, 2])[-1])
        padding_x = img_res - (x[1] - x[0] + 1)
        padding_y = img_res - (y[1] - y[0] + 1)
        x_padded = [x[0] - padding_x // 2, x[1] + padding_x // 2]
        y_padded = [y[0] - padding_y // 2, y[1] + padding_y // 2]

        if padding_x % 2 != 0:
            x_padded[1] += 1

        if padding_y % 2 != 0:
            y_padded[1] += 1

        if x_padded[1] > img_dims[0] - 1:
            x_padded[0] -= (x_padded[1] - img_dims[0] + 1)
            x_padded[1] -= (x_padded[1] - img_dims[0] + 1)
        elif x_padded[0] < 0:
            x_padded[1] -= x_padded[0]
            x_padded[0] -= x_padded[0]
        else:
            pass

        if y_padded[1] > img_dims[1] - 1:
            y_padded[0] -= (y_padded[1] - img_dims[1] + 1)
            y_padded[1] -= (y_padded[1] - img_dims[1] + 1)
        elif y_padded[0] < 0:
            y_padded[1] += y_padded[0]
            y_padded[0] += y_padded[0]
        else:
            pass

        for i in range(*z):
            imgs.append(img_arr[x_padded[0]:(x_padded[1] + 1), y_padded[0]:(y_padded[1] + 1), i][:, :, np.newaxis])
            segs.append(seg_arr[x_padded[0]:(x_padded[1] + 1), y_padded[0]:(y_padded[1] + 1), i][:, :, np.newaxis])

    return imgs, segs


#-------------------------------------------------------------------------
""" Data augmentation layer """

class Augmentation(tf.keras.layers.Layer):
    def __init__(self, img_res, seed=5):
        super().__init__(name="augmentation")

        self.image_augment = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed),
            tf.keras.layers.RandomZoom(0.2, seed=seed),
            tf.keras.layers.RandomCrop(img_res[0], img_res[1], seed=seed)
        ]

        self.label_augment = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed, interpolation="nearest"),
            tf.keras.layers.RandomZoom(0.2, seed=seed, interpolation="nearest"),
            tf.keras.layers.RandomCrop(img_res[0], img_res[1], seed=seed)
        ]

    def call(self, x, y):
        for aug in self.image_augment:
            x = aug(x)
        
        for aug in self.label_augment:
            y = aug(y)

        return x, y


#-------------------------------------------------------------------------
""" Sorenson-Dice loss for segmentation-based evaluation """

def dice_loss(A, B, axis=None):
    numerator = 2 * tf.reduce_sum(A * B, axis=axis)
    denominator = tf.reduce_sum(A, axis=axis) + tf.reduce_sum(B, axis=axis) + 1e-12

    return 1 - numerator / denominator


#-------------------------------------------------------------------------
""" Custom callback for saving training and test images """

class ImgSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_ex, test_ex, save_path):
        super().__init__()
        self.train_examples = train_ex
        self.test_examples = test_ex
        self.save_path = save_path
        if not os.path.exists(f"{save_path}/train"): os.makedirs(f"{save_path}/train")
        if not os.path.exists(f"{save_path}/test"): os.mkdir(f"{save_path}/test")
        self.HU_min = -500
        self.HU_max = 2500
    
    def save_pred(self, examples, phase, epoch):
        imgs, segs = examples
        preds = np.round(self.model(imgs, training=False).numpy())
        imgs = imgs.numpy() * (self.HU_max - self.HU_min) + self.HU_min
        segs = segs.numpy()

        fig, axs = plt.subplots(imgs.shape[0], 5)

        for i in range(imgs.shape[0]):
            axs[i, 0].imshow(imgs[i, :, :, 0].T, cmap="gray", vmin=-150, vmax=250)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(preds[i, :, :, 0].T, cmap="hot")
            axs[i, 1].axis("off")
            axs[i, 2].imshow(segs[i, :, :, 0].T, cmap="hot")
            axs[i, 2].axis("off")
            axs[i, 3].imshow(imgs[i, :, :, 0].T, cmap="gray", vmin=-150, vmax=250)
            axs[i, 3].imshow(np.ma.masked_where(preds == 0.0, preds)[i, :, :, 0].T, cmap="Set1")
            axs[i, 3].axis("off")
            axs[i, 4].imshow(imgs[i, :, :, 0].T, cmap="gray", vmin=-150, vmax=250)
            axs[i, 4].imshow(np.ma.masked_where(segs == 0.0, segs)[i, :, :, 0].T, cmap="Set1")
            axs[i, 4].axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{phase}/{epoch}.png", dpi=250)
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.save_pred(self.train_examples, "train", epoch + 1)
            self.save_pred(self.test_examples, "test", epoch + 1)


#-------------------------------------------------------------------------

def bootstrap(sample1, sample2, N):
    median_diff = []
    n1 = sample1.shape[0]

    if sample2 is None:
        for _ in range(N):
            resampled1 = random.choices(sample1, k=n1)
            median_diff.append(np.median(resampled1))

    else:
        n2 = sample2.shape[0]

        for _ in range(N):
            resampled1 = random.choices(sample1, k=n1)
            resampled2 = random.choices(sample2, k=n2)
            median_diff.append(np.median(resampled1) - np.median(resampled2))

    return median_diff


#-------------------------------------------------------------------------
""" Routine for testing data loader, augmentation and Dice """

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    imgs, segs = load_data(
        "test",
        128,
        "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real/Images",
        "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real/Segmentations_Tumour",
        "VC",
        "T065A1"
    )

    ds = tf.data.Dataset.from_tensor_slices((imgs, segs)).shuffle(256).batch(4).map(Augmentation([128, 128]))

    for img, seg in ds:
        fig, axs = plt.subplots(2, 4)

        for i in range(4):
            print(dice_loss(seg[i, ...], seg[i, ...]))
            axs[0, i].imshow(img[i, ...], cmap="gray")
            axs[1, i].imshow(seg[i, ...], cmap="gray")

        plt.show()
