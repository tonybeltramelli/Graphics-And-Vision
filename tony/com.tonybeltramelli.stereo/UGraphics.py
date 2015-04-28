__author__ = 'tbeltramelli'

import cv2
from pylab import *


class UGraphics(object):

    @staticmethod
    def hex_color_to_bgr(hexadecimal):
        red = (hexadecimal & 0xFF0000) >> 16
        green = (hexadecimal & 0xFF00) >> 8
        blue = (hexadecimal & 0xFF)
        return [blue, green, red]

    @staticmethod
    def draw_vector(img, x, y, magnitude, orientation):
        c_x = int(np.cos(orientation) * magnitude)
        c_y = int(np.sin(orientation) * magnitude)

        cv2.arrowedLine(img, (x, y), (x + c_x, y + c_y), UGraphics.hex_color_to_bgr(0xf2f378), 1)

    @staticmethod
    def get_stereo_image(left_img, right_img, scale=1.0):
        height, width = left_img.shape[:2]
        width = int(width * scale)
        height = int(height * scale)

        left_img = cv2.resize(left_img, (width, height))
        right_img = cv2.resize(right_img, (width, height))

        image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        image[:height, :width] = left_img
        image[:height, width:width * 2] = right_img

        return image