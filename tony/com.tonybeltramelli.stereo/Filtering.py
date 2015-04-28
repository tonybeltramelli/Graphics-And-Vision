__author__ = 'tbeltramelli'

from pylab import *
import cv2


class Filtering(object):
    @staticmethod
    def get_gray_scale_image(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def apply_box_filter(image, n=3):
        box = np.ones((n, n), np.float32) / (n * n)

        return cv2.filter2D(image, -1, box)

    @staticmethod
    def apply_gaussian_blur(image, n=3):
        return cv2.GaussianBlur(image, (n, n), 0)

    @staticmethod
    def apply_blur(image, n=3):
        return cv2.blur(image, (n, n))