import abc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------------------


class AffineTransform(tf.keras.layers.Layer, abc.ABC):
    """Base class for performing affine transforms.
    Based on implementation of spatial transformer networks:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    _flat_coords: tf.Tensor
    _mb_size: int

    def __init__(self, img_dims: list[int], name: str):
        super().__init__(name=name, dtype="float32")

        if len(img_dims) == 2:
            self._depth = 1
            self._height = img_dims[0]
            self._width = img_dims[1]

        elif len(img_dims) == 3:
            self._depth = img_dims[0]
            self._height = img_dims[1]
            self._width = img_dims[2]

        else:
            raise ValueError(f"Invalid image dimensions: {img_dims}")

    @abc.abstractmethod
    def coord_gen(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform_coords(self, mb_size: int, thetas: tf.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_img_indices(self) -> tuple[tf.Tensor, tuple[tf.Tensor, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(
        self,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> list[tf.Tensor]:
        raise NotImplementedError

    def interpolate(
        self,
        im: tf.Tensor,
        base: tf.Tensor,
        weights: list[tf.Tensor],
        n_ch: int,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> tf.Tensor:
        raise NotImplementedError

    @property
    def flat_coords(self) -> tf.Tensor:
        return self._flat_coords

    def call(self, im: tf.Tensor, thetas: tf.Tensor) -> tf.Tensor:
        mb_size = tf.shape(im)[0]
        n_ch = tf.shape(im)[-1]
        self.transform_coords(mb_size, thetas)
        base, indices = self.get_img_indices()
        weights = self.get_weights(*indices)

        # If 2D image, add depth dim
        if self._depth == 1:
            im = im[:, tf.newaxis, :, :, :]
            # im = tf.reshape(
            #     im,
            #     [mb_size, self._depth, self._height, self._width, n_ch]
            # )

        output = self.interpolate(im, base, weights, n_ch, *indices)

        # If 2D image, remove depth dim
        if self._depth == 1:
            output = tf.reshape(output, [mb_size, self._height, self._width, n_ch])

        return output


# -------------------------------------------------------------------------


class AffineTransform2D(AffineTransform):
    """2D Affine transform class.
    Acts on 2D images and also depth-wise on 3D volumes
    """

    _X: tf.Tensor
    _Y: tf.Tensor

    def __init__(self, img_dims: list[int], name: str = "affine2D") -> None:
        super().__init__(img_dims, name=name)
        self.coord_gen()

    def coord_gen(self) -> None:
        """Generate flattened coordinates [3, height * width]."""
        # Coords in range [-1, 1] (assuming origin in centre)
        Y, X = tf.meshgrid(  # noqa: N806
            tf.linspace(-1.0, 1.0, self._width),
            tf.linspace(-1.0, 1.0, self._height),
            indexing="ij",
        )
        flat_X = tf.reshape(X, (1, -1))  # noqa: N806
        flat_Y = tf.reshape(Y, (1, -1))  # noqa: N806

        # Rows are X, Y and row of ones (row length is height * width)
        self._flat_coords = tf.concat(
            [flat_X, flat_Y, tf.ones((1, self._height * self._width))],
            axis=0,
        )

    def transform_coords(self, mb_size: int, thetas: tf.Tensor) -> None:
        """Transform flattened coordinates with transformation matrix theta.
        :param thetas: 6 params for transform [mb, 6]
        """
        self._mb_size = mb_size
        new_flat_coords = tf.tile(self._flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 2, 3])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self._X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self._Y = tf.reshape(new_flat_coords[:, 1, :], [-1])

    def get_img_indices(self) -> tuple[tf.Tensor, tuple[tf.Tensor, ...]]:
        """Generate base indices corresponding to each image in mb.
        E.g. [0   0   0
              hw  hw  hw
              2hw 2hw 2hw]

        where hw = height * width
        Allows selecting e.g. x, y pixel in second img in
        minibatch by selecting hw + x + y
        """
        # Convert coords to [0, width/height]
        self._X = (self._X + 1.0) / 2.0 * self._width
        self._Y = (self._Y + 1.0) / 2.0 * self._height

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self._X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self._Y), "int32")
        y1 = y0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self._width - 1)
        x1 = tf.clip_by_value(x1, 0, self._width - 1)
        y0 = tf.clip_by_value(y0, 0, self._height - 1)
        y1 = tf.clip_by_value(y1, 0, self._height - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(
            tf.range(self._mb_size) * self._height * self._width,
            [-1, 1],
        )
        img_indices = tf.matmul(
            img_indices,
            tf.ones((1, self._height * self._width), dtype="int32"),
        )
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1)

    def get_weights(
        self,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> list[tf.Tensor]:
        """Generate weights.
        These represent how close bracketing indices are to transformed coords
        """
        x0_f = tf.cast(x0, "float32")
        x1_f = tf.cast(x1, "float32")
        y0_f = tf.cast(y0, "float32")
        y1_f = tf.cast(y1, "float32")

        wa = tf.expand_dims((x1_f - self._X) * (y1_f - self._Y), 1)
        wb = tf.expand_dims((x1_f - self._X) * (self._Y - y0_f), 1)
        wc = tf.expand_dims((self._X - x0_f) * (y1_f - self._Y), 1)
        wd = tf.expand_dims((self._X - x0_f) * (self._Y - y0_f), 1)

        return [wa, wb, wc, wd]

    def interpolate(
        self,
        im: tf.Tensor,
        base: tf.Tensor,
        weights: list[tf.Tensor],
        n_ch: int,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> tf.Tensor:
        """Perform interpolation of minibatch of images."""
        # Add base image indices to the integer indices bracketing
        # the transformed coordinates
        indices = []
        indices.append(base + y0 * self._width + x0)
        indices.append(base + y1 * self._width + x0)
        indices.append(base + y0 * self._width + x1)
        indices.append(base + y1 * self._width + x1)

        # Get images using bracketed indices and take weighted average
        # Permute axes so that we perform same operation depth- and channel-wise
        # [N, D, H, W, C] -> [N, H, W, D, C]
        im = tf.transpose(im, [0, 2, 3, 1, 4])
        im_flat = tf.reshape(im, [-1, self._depth * n_ch])
        imgs = [tf.gather(im_flat, idx) for idx in indices]

        weighted_img = tf.add_n([img * weight for img, weight in zip(imgs, weights)])

        weighted_img = tf.reshape(
            weighted_img,
            [self._mb_size, self._height, self._width, self._depth, n_ch],
        )

        # [N, H, W, D, C] -> [N, D, H, W, C]
        weighted_img = tf.transpose(weighted_img, [0, 3, 1, 2, 4])

        return weighted_img

    @property
    def mesh_coords(self) -> tuple[tf.Tensor, tf.Tensor]:
        return self._Y, self._X


# -------------------------------------------------------------------------


class AffineTransform3D(AffineTransform):
    """3D affine transform class, not yet functional."""

    _X: tf.Tensor
    _Y: tf.Tensor
    _Z: tf.Tensor

    def __init__(self, img_dims: list[int], name: str = "affine3D") -> None:
        super().__init__(img_dims, name=name)
        assert len(img_dims) == 3, f"Invalid image dimensions: {img_dims}"
        self.depth_f = tf.cast(self.depth_i, tf.float32)
        self.coord_gen()
        raise NotImplementedError

    def coord_gen(self) -> None:
        """Generate flattened coordinates [4, height * width * depth]."""
        X, Y, Z = tf.meshgrid(  # noqa: N806
            tf.linspace(-1.0, 1.0, self.width_i),
            tf.linspace(-1.0, 1.0, self._height),
            tf.linspace(-1.0, 1.0, self._depth),
        )
        flat_X = tf.reshape(X, (1, -1))  # noqa: N806
        flat_Y = tf.reshape(Y, (1, -1))  # noqa: N806
        flat_Z = tf.reshape(Z, (1, -1))  # noqa: N806

        # Rows are X, Y, Z and row of ones (row length is height * width * depth)
        self._flat_coords = tf.concat(
            [
                flat_X,
                flat_Y,
                flat_Z,
                tf.ones((1, self._height * self._width * self._depth)),
            ],
            axis=0,
        )

    def transform_coords(self, mb_size: int, thetas: tf.Tensor) -> None:
        """Transform flattened coordinates with transformation matrix.
        :param thetas: 12 params for transform [mb, 12]
        """
        self._mb_size = mb_size
        new_flat_coords = tf.tile(self._flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 3, 4])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y/Z coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self._X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self._Y = tf.reshape(new_flat_coords[:, 1, :], [-1])
        self._Z = tf.reshape(new_flat_coords[:, 2, :], [-1])

    def get_img_indices(self) -> tuple[tf.Tensor, tuple[tf.Tensor, ...]]:
        """Generate base indices corresponding to each image in mb.
        E.g. [0    0    0
              hwd  hwd  hwd
              2hwd 2hwd 2hwd]

        where hw = height * width * depth
        Allows selecting e.g. x, y pixel in second img in
        minibatch by selecting hw + x + y + z
        """
        # Convert coords to [0, width/height/depth]
        self._X = (self._X + 1.0) / 2.0 * self._width
        self._Y = (self._Y + 1.0) / 2.0 * self._height
        self._Z = (self._Z + 1.0) / 2.0 * self._depth

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self._X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self._Y), "int32")
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(self._Z), "int32")
        z1 = z0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self._width - 1)
        x1 = tf.clip_by_value(x1, 0, self._width - 1)
        y0 = tf.clip_by_value(y0, 0, self._height - 1)
        y1 = tf.clip_by_value(y1, 0, self._height - 1)
        z0 = tf.clip_by_value(z0, 0, self._depth - 1)
        z1 = tf.clip_by_value(z1, 0, self._depth - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(
            tf.range(self._mb_size) * self._height * self._width * self._depth,
            [-1, 1],
        )
        img_indices = tf.matmul(
            img_indices,
            tf.ones((1, self._height * self._width * self._depth), dtype="int32"),
        )
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1, z0, z1)

    def get_weights(
        self,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> list[tf.Tensor]:
        """Generate weights.
        These represent how close bracketing indices are to transformed coords
        """
        raise NotImplementedError

    def interpolate(
        self,
        im: tf.Tensor,
        base: tf.Tensor,
        weights: list[tf.Tensor],
        n_ch: int,
        x0: tf.Tensor,
        x1: tf.Tensor,
        y0: tf.Tensor,
        y1: tf.Tensor,
    ) -> tf.Tensor:
        raise NotImplementedError

    @property
    def mesh_coords(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self._Z, self._Y, self._X


# -------------------------------------------------------------------------


if __name__ == "__main__":
    """Routine for visually testing implementations"""
    # Test 2D image
    img_vol = np.zeros([4, 128, 128, 1])
    img_vol[:, 54:74, 54:74, :] = 1

    theta0 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype="float32")
    theta1 = np.array([2, 0, 0, 0, 2, 0], dtype="float32")
    theta2 = np.array([1, 0, -0.5, 0, 1, 0.25], dtype="float32")
    theta3 = np.array([0.707, -0.707, 0.5, 0.707, 0.707, 0.25], dtype="float32")

    theta = tf.convert_to_tensor(np.stack([theta0, theta1, theta2, theta3], axis=0))
    AT = AffineTransform2D([128, 128])
    new_vol = AT(img_vol, theta)

    fig, axs = plt.subplots(2, 4)

    for j in range(4):
        axs[0, j].imshow(img_vol[j, :, :, 0])
        axs[1, j].imshow(new_vol[j, :, :, 0])

    plt.show()

    # Test 2D transforms on 3D image volumes
    img_vol = np.zeros([4, 64, 128, 128, 1])
    img_vol[:, :, 54:74, 54:74, :] = 1

    theta0 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype="float32")
    theta1 = np.array([2, 0, 0, 0, 2, 0], dtype="float32")
    theta2 = np.array([1, 0, -0.5, 0, 1, 0.25], dtype="float32")
    theta3 = np.array([0.707, -0.707, 0.5, 0.707, 0.707, 0.25], dtype="float32")

    theta = tf.convert_to_tensor(np.stack([theta0, theta1, theta2, theta3], axis=0))
    AT = AffineTransform2D([64, 128, 128])
    new_vol = AT(img_vol, theta)

    for i in range(0, 12, 2):
        fig, axs = plt.subplots(2, 4)

        for j in range(4):
            axs[0, j].imshow(img_vol[j, i, :, :, 0])
            axs[1, j].imshow(new_vol[j, i, :, :, 0])

        plt.show()
