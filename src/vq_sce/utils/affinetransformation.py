import abc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


""" Based on implementation of spatial transformer networks:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
"""

#-------------------------------------------------------------------------

""" Abstract base class """
class AffineTransform(tf.keras.layers.Layer, abc.ABC):

    def __init__(self, img_dims: list, name: str):
        super().__init__(name=name)
        self.num_dims = len(img_dims)
        self.height_i = img_dims[0]
        self.width_i = img_dims[1]
        self.height_f = tf.cast(self.height_i, "float32")
        self.width_f = tf.cast(self.width_i, "float32")

        if self.num_dims == 3:
            self.depth_i = 1
            # self.n_ch = img_dims[2]
        
        elif self.num_dims == 4:
            self.depth_i = img_dims[2]
            # self.n_ch = img_dims[3]
        
        else:
            raise ValueError(f"Invalid image dimensions: {img_dims}")

        self.flat_coords = None

    @abc.abstractmethod
    def coord_gen(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def transform_coords(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_img_indices(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_weights(self):
        raise NotImplementedError
    
    def interpolate(self):
        raise NotImplementedError

    def call(self, im: object, mb_size: int, thetas: object) -> object:
        n_ch = im.shape[4]
        self.transform_coords(mb_size, thetas)
        base, indices = self.get_img_indices()
        weights = self.get_weights(*indices)
        output = self.interpolate(im, base, weights, n_ch, *indices)

        if self.num_dims == 3:
            return tf.reshape(output, [mb_size, self.height_i, self.width_i, n_ch])
        else:
            return tf.reshape(output, [mb_size, self.height_i, self.width_i, self.depth_i, n_ch])


#-------------------------------------------------------------------------
""" 2D affine transform class, acts on 2D images and also
    depth-wise on 3D volumes """

class AffineTransform2D(AffineTransform):

    def __init__(self, img_dims: list, name: str = "affine2D"):
        super().__init__(img_dims, name=name)
        self.mb_size = None
        self.X, self.Y = None, None
        self.coord_gen()

    def coord_gen(self) -> None:
        """ Generate flattened coordinates [3, height * width] """

        # Coords in range [-1, 1] (assuming origin in centre)
        X, Y = tf.meshgrid(tf.linspace(-1.0, 1.0, self.width_i), tf.linspace(-1.0, 1.0, self.height_i))
        flat_X = tf.reshape(X, (1, -1))
        flat_Y = tf.reshape(Y, (1, -1))

        # Rows are X, Y and row of ones (row length is height * width)
        self.flat_coords = tf.concat([flat_X, flat_Y, tf.ones((1, self.height_i * self.width_i))], axis=0)

    def transform_coords(self, mb_size: int, thetas: object) -> None:
        """ Transform flattened coordinates with transformation matrix theta
            thetas: 6 params for transform [mb, 6] """

        self.mb_size = mb_size
        new_flat_coords = tf.tile(self.flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 2, 3])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self.X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self.Y = tf.reshape(new_flat_coords[:, 1, :], [-1])
    
    def get_img_indices(self) -> tuple:
        """ Generates base indices corresponding to each image in mb
            e.g. [0   0   0
                  hw  hw  hw
                  2hw 2hw 2hw]

            where hw = height * width
            Allows selecting e.g. x, y pixel in second img in minibatch by selecting hw + x + y """

        # Convert coords to [0, width/height]
        self.X = (self.X + 1.0) / 2.0 * (self.width_f)
        self.Y = (self.Y + 1.0) / 2.0 * (self.height_f)

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self.X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self.Y), "int32")
        y1 = y0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self.width_i - 1)
        x1 = tf.clip_by_value(x1, 0, self.width_i - 1)
        y0 = tf.clip_by_value(y0, 0, self.height_i - 1)
        y1 = tf.clip_by_value(y1, 0, self.height_i - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(tf.range(self.mb_size) * self.height_i * self.width_i, [-1, 1])
        img_indices = tf.matmul(img_indices, tf.ones((1, self.height_i * self.width_i), dtype="int32"))
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1)
    
    def get_weights(self, x0, x1, y0, y1) -> list:
        """ Generate weights representing how close bracketing indices are to transformed coords """

        x0_f = tf.cast(x0, "float32")
        x1_f = tf.cast(x1, "float32")
        y0_f = tf.cast(y0, "float32")
        y1_f = tf.cast(y1, "float32")

        wa = tf.expand_dims((x1_f - self.X) * (y1_f - self.Y), 1)
        wb = tf.expand_dims((x1_f - self.X) * (self.Y - y0_f), 1)
        wc = tf.expand_dims((self.X - x0_f) * (y1_f - self.Y), 1)
        wd = tf.expand_dims((self.X - x0_f) * (self.Y - y0_f), 1)

        return [wa, wb, wc, wd]

    def interpolate(self, im: object, base: object, weights: list, n_ch: int, x0, x1, y0, y1) -> object:
        """ Perform interpolation of minibatch of images """

        # Add base image indices to the integer indices bracketing the transformed coordinates
        indices = []
        indices.append(base + y0 * self.width_i + x0)
        indices.append(base + y1 * self.width_i + x0)
        indices.append(base + y0 * self.width_i + x1)
        indices.append(base + y1 * self.width_i + x1)

        # Get images using bracketed indices and take weighted average
        im_flat = tf.reshape(im, [-1, self.depth_i * n_ch])
        imgs = [tf.gather(im_flat, idx) for idx in indices]

        return tf.add_n([img * weight for img, weight in zip(imgs, weights)])

#-------------------------------------------------------------------------
""" 3D affine transform class, not yet functional """

class AffineTransform3D(AffineTransform):

    def __init__(self, img_dims: list, name: str = "affine2D"):
        super().__init__(img_dims, name=name)
        assert self.num_dims == 4, f"Invalid image dimensions: {self.num_dims}"
        self.depth_f = tf.cast(self.depth_i, tf.float32)
        self.mb_size = None
        self.X, self.Y, self.Z = None, None, None
        self.coord_gen()
        raise NotImplementedError
    
    def coord_gen(self) -> None:
        """ Generate flattened coordinates [4, height * width * depth] """

        X, Y, Z = tf.meshgrid(tf.linspace(-1.0, 1.0, self.width_i), tf.linspace(-1.0, 1.0, self.height_i), tf.linspace(-1.0, 1.0, self.depth_i))
        flat_X = tf.reshape(X, (1, -1))
        flat_Y = tf.reshape(Y, (1, -1))
        flat_Z = tf.reshape(Z, (1, -1))

        # Rows are X, Y, Z and row of ones (row length is height * width * depth)
        self.flat_coords = tf.concat([flat_X, flat_Y, flat_Z, tf.ones((1, self.height_i * self.width_i * self.depth_i))], axis=0)
    
    def transform_coords(self, mb_size: int, thetas: object) -> None:
        """ Transform flattened coordinates with transformation matrix
            thetas: 12 params for transform [mb, 12] """

        self.mb_size = mb_size
        new_flat_coords = tf.tile(self.flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 3, 4])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y/Z coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self.X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self.Y = tf.reshape(new_flat_coords[:, 1, :], [-1])
        self.Z = tf.reshape(new_flat_coords[:, 2, :], [-1])

    def get_img_indices(self) -> tuple:
        """ Generates base indices corresponding to each image in mb
            e.g. [0    0    0
                  hwd  hwd  hwd
                  2hwd 2hwd 2hwd]

            where hw = height * width * depth
            Allows selecting e.g. x, y pixel in second img in minibatch by selecting hw + x + y + z"""

        # Convert coords to [0, width/height/depth]
        self.X = (self.X + 1.0) / 2.0 * (self.width_f)
        self.Y = (self.Y + 1.0) / 2.0 * (self.height_f)
        self.Z = (self.Z + 1.0) / 2.0 * (self.depth_f)

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self.X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self.Y), "int32")
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(self.Z), "int32")
        z1 = z0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self.width_i - 1)
        x1 = tf.clip_by_value(x1, 0, self.width_i - 1)
        y0 = tf.clip_by_value(y0, 0, self.height_i - 1)
        y1 = tf.clip_by_value(y1, 0, self.height_i - 1)
        z0 = tf.clip_by_value(z0, 0, self.depth_i - 1)
        z1 = tf.clip_by_value(z1, 0, self.depth_i - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(tf.range(self.mb_size) * self.height_i * self.width_i * self.depth_i, [-1, 1])
        img_indices = tf.matmul(img_indices, tf.ones((1, self.height_i * self.width_i * self.depth_i), dtype="int32"))
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1, z0, z1)
    
    def get_weights(self):
        """ Generate weights representing how close bracketing indices are to transformed coords """
        return super().get_weights()
    
    def interpolate(self, im, base, weights, x0, x1, y0, y1, z0, z1):
        return super().interpolate()


#-------------------------------------------------------------------------
""" Routine for visually testing implementations """

if __name__ == "__main__":

    img_vol = np.zeros([4, 128, 128, 64, 1])
    img_vol[:, 54:74, 54:74, :, :] = 1

    theta0 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype="float32")
    theta1 = np.array([2, 0, 0, 0, 2, 0], dtype="float32")
    theta2 = np.array([1, 0, -0.5, 0, 1, 0.25], dtype="float32")
    theta3 = np.array([0.707, -0.707, 0.5, 0.707, 0.707, 0.25], dtype="float32")

    theta = tf.convert_to_tensor(np.stack([theta0, theta1, theta2, theta3], axis=0))
    AT = AffineTransform2D([128, 128, 64, 1])

    new_vol = AT(img_vol, 4, theta)

    for i in range(0, 12, 2):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(img_vol[0, :, :, i, 0])
        axs[0, 1].imshow(img_vol[1, :, :, i, 0])
        axs[0, 2].imshow(img_vol[2, :, :, i, 0])
        axs[0, 3].imshow(img_vol[3, :, :, i, 0])
        axs[1, 0].imshow(new_vol[0, :, :, i, 0])
        axs[1, 1].imshow(new_vol[1, :, :, i, 0])
        axs[1, 2].imshow(new_vol[2, :, :, i, 0])
        axs[1, 3].imshow(new_vol[3, :, :, i, 0])
        plt.show()
