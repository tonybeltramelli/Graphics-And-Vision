__author__ = 'tbeltramelli'

import cv2
from UInteractive import *

class AHomography:
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
        print(points)
        print("-----")
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