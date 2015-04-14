__author__ = 'tbeltramelli'

import numpy as np
import math
from pylab import *

class UMath:

    @staticmethod
    def normalize(range_min, range_max, x, x_min, x_max):
        return range_min + (((x - x_min) * (range_max - range_min)) / (x_max - x_min))

    @staticmethod
    def is_in_area(x, y, width, height):
        return ((x > width/4) and (x < 3 * (width/4))) and ((y > height/4) and (y < 3 * (height/4)))

    @staticmethod
    def get_circle_samples(center=(0, 0), radius=1, point_number=30):
        s = np.linspace(0, 2 * math.pi, point_number)
        return [(radius * np.cos(t) + center[0], radius * np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s]

    @staticmethod
    def get_line_coordinates(p1, p2):
        (x1, y1) = p1
        x1 = int(x1)
        y1 = int(y1)

        (x2, y2) = p2
        x2 = int(x2)
        y2 = int(y2)

        points = []
        is_steep = abs(y2 - y1) > abs(x2 - x1)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        to_reverse = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            to_reverse = True
        delta_x = x2 - x1
        delta_y = abs(y2 - y1)
        error = int(delta_x / 2)
        y = y1
        y_step = None
        if y1 < y2:
            y_step = 1
        else:
            y_step = -1
        for x in range(x1, x2 + 1):
            if is_steep:
                points.append([y, x])
            else:
                points.append([x, y])
            error -= delta_y
            if error < 0:
                y += y_step
                error += delta_x
        if to_reverse:
            points.reverse()

        result = np.array(points)
        all_x = result[:, 0]
        all_y = result[:, 1]

        return result

    @staticmethod
    def to_homogenious(points):
        return vstack((points, ones((1, points.shape[1]))))

    @staticmethod
    def get_rotation_translation_matrix(rotation_factor):
        rotation_1, rotation_2, translation = tuple(np.hsplit(rotation_factor, 3))
        rotation_3 = cross(rotation_1.T, rotation_2.T).T

        return np.hstack((rotation_1, rotation_2, rotation_3, translation))