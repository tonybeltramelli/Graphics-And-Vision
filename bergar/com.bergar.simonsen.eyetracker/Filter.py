__author__ = 'bs'

from pylab import *
import cv2

class Filter:
    @staticmethod
    def gaussianBlur(image, n=3):
        return cv2.GaussianBlur(image, (n, n), 0)

    @staticmethod
    def blur(image, n=3):
        return cv2.blur(image, (n, n))