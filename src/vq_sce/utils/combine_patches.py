import numpy as np
import tensorflow as tf

from .patch_utils import generate_indices


#-------------------------------------------------------------------------

class CombinePatches:
    def __init__(self, config: dict):
        self.config = config
        self.linear_weights = None
        self.linear_AC = None
        self.linear_VC = None
        self.single_coords = None
        self.stride_length = config["data"]["stride_length"]

    def new_subject(self, subject_dims: list):
        self.HWD_dims = subject_dims

        self.DHW_dims = [self.HWD_dims[2], self.HWD_dims[0], self.HWD_dims[1]]
        self.linear_img_size = tf.reduce_prod(self.DHW_dims)
        self.linear_AC = tf.zeros(self.linear_img_size, "int16")
        self.linear_VC = tf.zeros(self.linear_img_size, "int16")
        self.linear_weights = np.zeros(self.linear_img_size, "int16")

        # Need linear coords for our (HWD) dim order
        self.linear_coords = tf.reshape(tf.range(self.linear_img_size), self.HWD_dims)

    def get_AC(self):
        linear_AC = tf.cast(tf.round(self.linear_AC / self.linear_weights), "int16")
        AC = tf.reshape(linear_AC, self.HWD_dims)

        return AC.numpy()

    def get_VC(self):
        linear_VC = tf.cast(tf.round(self.linear_VC / self.linear_weights), "int16")
        VC = tf.reshape(linear_VC, self.HWD_dims)

        return VC.numpy()

    def reset(self):
        self.linear_AC = tf.zeros(self.linear_img_size, "int16")
        self.linear_VC = tf.zeros(self.linear_img_size, "int16")
        self.linear_weights = np.zeros(self.linear_img_size, "int16")

    def apply_patches(self, AC, VC, coords):
        # Flatten minibatch of linear coords
        coords = tf.reshape(coords, [-1, 1])

        if AC is not None:
            # Flatten into update vector
            AC_update = tf.cast(tf.round(tf.reshape(AC, -1)), "int16")
            # Update 1D image with patches
            self.linear_AC = tf.tensor_scatter_nd_add(self.linear_AC, coords, AC_update)

        if VC is not None:
            VC_update = tf.cast(tf.round(tf.reshape(VC, -1)), "int16")
            self.linear_VC = tf.tensor_scatter_nd_add(self.linear_VC, coords, VC_update)

        # Update weights
        self.linear_weights = tf.tensor_scatter_nd_add(self.linear_weights, coords, tf.ones_like(AC_update))


#-------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import tensorflow as tf

    """ Quick routine to visually check output of CombinePatches """

    # stride = 2
    # patch_size = [2, 2, 2]
    # test_config = {"data": {"stride_length": stride}}

    # Combine = CombinePatches(test_config)

    # im = np.zeros((8, 8, 5))
    # im[0:4, 0:4, 0:2] = 1
    # im[-4:, -4:, 0:2] = 1
    # im[0:4, -4:, -2:] = 1
    # im[-4:, 0:4, -2:] = 1

    stride = 3
    patch_size = [8, 8, 3]
    test_config = {"data": {"stride_length": stride}}

    Combine = CombinePatches(test_config)

    im = np.zeros((64, 64, 31))
    im[0:32, 0:32, 0:16] = 1
    im[-32:, -32:, 0:16] = 1
    im[0:32, -32:, -16:] = 1
    im[-32:, 0:32, -16:] = 1

    indices = generate_indices(im, stride_length=stride, patch_size=patch_size, downsample=1)

    linear_img = np.reshape(im, -1)
    linear_img_size = tf.reduce_prod(linear_img.shape)
    single_coords = tf.reshape(tf.range(linear_img_size), im.shape)
    patches_to_stack = []

    for index in indices:
        patch = tf.reshape(tf.gather(linear_img, index), patch_size)
        patches_to_stack.append(patch.numpy())


    AC_mb = np.squeeze(np.stack(patches_to_stack, axis=0)).astype("int16")
    VC_mb = 1 - AC_mb
    idx_mb = tf.stack(indices, axis=0)
    print(idx_mb.shape, AC_mb.shape)
    Combine.new_subject(im.shape)
    print(Combine.linear_AC.shape)
    Combine.apply_patches(AC_mb, VC_mb, idx_mb)
    AC = Combine.get_AC()
    VC = Combine.get_VC()

    plt.imshow(AC[:, :, 0])
    plt.show()
    plt.imshow(AC[:, :, -1])
    plt.show()
    plt.imshow(VC[:, :, 0])
    plt.show()
    plt.imshow(VC[:, :, -1])
    plt.show()
