import numpy as np
import tensorflow as tf

from .patch_utils import generate_indices


#-------------------------------------------------------------------------

class CombinePatches:
    def __init__(self, stride_length: int) -> None:
        self.linear_weights = None
        self.linear = None
        self.single_coords = None
        self.stride_length = stride_length

    def new_subject(self, subject_dims: list) -> None:
        self.DHW_dims = subject_dims
        self.linear_img_size = tf.reduce_prod(self.DHW_dims)
        self.linear = tf.zeros(self.linear_img_size, "int16")
        self.linear_weights = np.zeros(self.linear_img_size, "int16")

        # Need linear coords for our (HWD) dim order
        self.linear_coords = tf.reshape(tf.range(self.linear_img_size), self.DHW_dims)

    def get_img(self) -> np.ndarray:
        linear = tf.cast(tf.round(self.linear / self.linear_weights), "int16")
        img = tf.reshape(linear, self.DHW_dims)

        return img.numpy()

    def reset(self) -> None:
        self.linear = tf.zeros(self.linear_img_size, "int16")
        self.linear_weights = np.zeros(self.linear_img_size, "int16")

    def apply_patches(self, patches, coords) -> None:
        # Flatten minibatch of linear coords
        coords = tf.reshape(coords, [-1, 1])

        update = tf.cast(tf.round(tf.reshape(patches, -1)), "int16")
        # Update 1D image with patches
        self.linear = tf.tensor_scatter_nd_add(self.linear, coords, update)

        # Update weights
        self.linear_weights = tf.tensor_scatter_nd_add(self.linear_weights, coords, tf.ones_like(update))


#-------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import tensorflow as tf

    """ Quick routine to visually check output of CombinePatches """

    stride = 2
    patch_size = [2, 2, 2]
    Combine = CombinePatches(stride)

    im = np.zeros((4, 8, 8))
    im[0:2, 0:4, 0:4] = 1
    im[0:2, -4:, -4:] = 1
    im[-2:, -4:, 0:4] = 1
    im[-2:, 0:4, -4:] = 1

    # stride = 8
    # patch_size = [16, 16, 16]
    # Combine = CombinePatches(stride)

    # im = np.zeros((32, 64, 64))
    # im[0:16, 0:32, 0:32] = 1
    # im[0:16, -32:, -32:] = 1
    # im[-16:, 0:32, -32:] = 1
    # im[-16:, -32:, 0:32] = 1

    indices = generate_indices(im, stride_length=stride, patch_size=patch_size)
    linear_img = np.reshape(im, -1)
    linear_img_size = tf.reduce_prod(linear_img.shape)
    single_coords = tf.reshape(tf.range(linear_img_size), im.shape)
    patches_to_stack = []

    for index in indices:
        patch = tf.reshape(tf.gather(linear_img, index), patch_size)
        patches_to_stack.append(patch.numpy())

    mb = np.squeeze(np.stack(patches_to_stack, axis=0)).astype("int16")

    idx_mb = tf.stack(indices, axis=0)
    print(idx_mb.shape, mb.shape)
    Combine.new_subject(im.shape)
    Combine.apply_patches(mb, idx_mb)
    img = Combine.get_img()

    plt.imshow(img[0, :, :])
    plt.show()
    plt.imshow(img[-1, :, :])
    plt.show()
