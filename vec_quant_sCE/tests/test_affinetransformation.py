import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from syntheticcontrast_v02.utils.affinetransformation import AffineTransform2D


class TestAffine2D(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.theta1 = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        self.theta2 = np.array([[1, 0, -1], [0, 1, 2]], dtype="float32")
        self.theta3 = np.array([[1, 5, -1], [3, 1, 2]], dtype="float32")
    
    def tearDown(self):
        super().tearDown()
    
    def test_coord_gen(self):
        AF = AffineTransform2D((2, 3, 1))
        self.assertAllEqual(AF.flat_coords.shape, (3, 6))
    
    def test_transform_coords(self):
        AF = AffineTransform2D((2, 3, 1))
        flat_coords = AF.flat_coords.numpy()
               
        new_coords1 = self.theta1 @ flat_coords
        new_coords2 = self.theta2 @ flat_coords
        new_coords3 = self.theta3 @ flat_coords
        AF.transform_coords(3, tf.concat([self.theta1.ravel(), self.theta2.ravel(), self.theta3.ravel()], axis=0))

        self.assertAllEqual(AF.X, np.hstack([new_coords1[0, :], new_coords2[0, :], new_coords3[0, :]]))
        self.assertAllEqual(AF.Y, np.hstack([new_coords1[1, :], new_coords2[1, :], new_coords3[1, :]]))
    
    def test_get_img_indices(self):
        AF = AffineTransform2D((2, 3, 1))
        AF.transform_coords(3, tf.concat([self.theta1.ravel(), self.theta2.ravel(), self.theta3.ravel()], axis=0))
        gt = np.hstack([[6 * mb] * 6 for mb in range(3)])
        base, _ = AF.get_img_indices()

        self.assertAllEqual(base, gt)

if __name__ == "__main__":

    """ Quick routine to visually check output of transforms """

    img_vol = np.zeros((4, 64, 64, 12, 1), dtype="float32")
    img_vol[:, 24:40, 24:40, :, :] = 1

    theta0 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype=np.float32)
    theta1 = np.array([2, 0, 0, 0, 2, 0], dtype=np.float32)
    theta2 = np.array([1, 0, -0.5, 0, 1, 0.25], dtype=np.float32)
    theta3 = np.array([0.707, -0.707, 0, 0.707, 0.707, 0], dtype=np.float32)

    theta = tf.convert_to_tensor(np.stack([theta0, theta1, theta2, theta3], axis=0))

    start_t = time.time()
    AT = AffineTransform2D((64, 64, 12, 1))
    new_vol = AT(img_vol, 4, theta)

    print(time.time() - start_t)

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(img_vol[0, :, :, 5, 0])
    axs[0, 1].imshow(img_vol[1, :, :, 5, 0])
    axs[0, 2].imshow(img_vol[2, :, :, 5, 0])
    axs[0, 3].imshow(img_vol[3, :, :, 5, 0])
    axs[1, 0].imshow(new_vol[0, :, :, 5, 0])
    axs[1, 1].imshow(new_vol[1, :, :, 5, 0])
    axs[1, 2].imshow(new_vol[2, :, :, 5, 0])
    axs[1, 3].imshow(new_vol[3, :, :, 5, 0])
    plt.show()