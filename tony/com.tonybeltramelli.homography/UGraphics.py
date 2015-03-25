__author__ = 'tbeltramelli'

import cv2
from pylab import *


class UGraphics:

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