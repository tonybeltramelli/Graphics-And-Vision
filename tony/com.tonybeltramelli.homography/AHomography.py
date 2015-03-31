__author__ = 'tbeltramelli'

import cv2
from pylab import *


class AHomography:
    _homography = None

    def build_homography(self, images):
        if self._homography is None:
            self.build_homography_from_mouse(images)

    def build_homography_from_mouse(self, images, n=4):
        image_points = []
        fig = figure(1)

        for i, img in enumerate(images):
            img = copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax = subplot(1, 2, i + 1)
            ax.imshow(img)

            title("Select " + str(n) + " points.")

            fig.canvas.draw()
            ax.hold('On')

            image_points.append(fig.ginput(n, -1))

            for p in image_points[i]:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)

            ax.imshow(img)

        close(fig)

        points1 = np.array([[x, y] for (x, y) in image_points[0]])
        points2 = np.array([[x, y] for (x, y) in image_points[len(image_points) - 1]])

        self._homography, mask = cv2.findHomography(points1, points2)

    def build_homography_from_coordinates(self, img, points):
        height, width, layers = img.shape

        points1 = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        points2 = np.array([[x, y] for (x, y) in points])

        self._homography, mask = cv2.findHomography(points1, points2)

    def get_mapped_texture(self, texture, background):
        height, width, layers = background.shape
        overlay = cv2.warpPerspective(texture, self._homography, (width, height))

        return cv2.addWeighted(background, 0.7, overlay, 0.3, 0)

    def get_2d_transform(self, x, y):
        a = array([[x], [y], [1]])

        result = np.dot(self._homography, a)

        x = result[0] / result[2]
        y = result[1] / result[2]

        return x, y