__author__ = 'tbeltramelli'

from UInteractive import *


class AHomography:
    _homography_output_path = None
    _homography = None

    def get_homography_all_from_mouse(self, images, n=4):
        image_points = UInteractive.select_points_in_images(images, n)

        points1 = np.array([[x, y] for (x, y) in image_points[0]])
        points2 = np.array([[x, y] for (x, y) in image_points[len(image_points) - 1]])

        homography, mask = cv2.findHomography(points1, points2)

        return homography

    def get_homography_part_from_mouse(self, img1, img2):
        points1 = self.get_points_from_image(img1)

        image_points = UInteractive.select_points_in_images([img2], 4)
        points2 = np.array([[x, y] for (x, y) in image_points[0]])

        homography, mask = cv2.findHomography(points1, points2)

        return homography

    def get_homography_from_coordinates(self, img, points):
        points1 = self.get_points_from_image(img)
        points2 = np.array([[x, y] for (x, y) in points])

        homography, mask = cv2.findHomography(points1, points2)

        return homography

    def get_mapped_texture(self, texture, background, homography):
        height, width, layers = background.shape
        overlay = cv2.warpPerspective(texture, homography, (width, height))

        return cv2.addWeighted(background, 0.7, overlay, 0.3, 0)

    def get_2d_transform_from_homography(self, x, y, homography):
        a = array([[x], [y], [1]])

        result = np.dot(homography, a)

        x = int(result[0] / result[2])
        y = int(result[1] / result[2])

        return x, y

    def get_points_from_image(self, img):
        height, width, layers = img.shape

        return np.array([[0, 0], [width, 0], [0, height], [width, height]])

    def define_map_homography(self, images):
        if self._homography is None:
            try:
                self._homography = np.load(self._homography_output_path + ".npy")
            except IOError, e:
                self._homography = self.get_homography_all_from_mouse(images)
                np.save(self._homography_output_path, self._homography)

    @staticmethod
    def estimate_homography(points1, points2):
        if len(points1) == 4 and len(points2) == 4:
            x1, y1 = points1[0]
            x2, y2 = points1[1]
            x3, y3 = points1[2]
            x4, y4 = points1[3]

            x_1, y_1 = points2[0]
            x_2, y_2 = points2[1]
            x_3, y_3 = points2[2]
            x_4, y_4 = points2[3]

            a = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
                           [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
                           [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
                           [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
                           [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
                           [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
                           [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_4, x_4],
                           [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_4, y_4]])

            u, d, v = np.linalg.svd(a)

            h = v[8]

            homography = np.matrix([[h[0, 0], h[0, 1], h[0, 2]],
                                    [h[0, 3], h[0, 4], h[0, 5]],
                                    [h[0, 6], h[0, 7], h[0, 8]]])

            homography = homography / homography[2, 2]
            return homography