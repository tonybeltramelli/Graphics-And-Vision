__author__ = 'tbeltramelli'

from pylab import *


class Camera:
    camera_matrix = np.zeros((3, 3))
    distortion_coefficient = np.zeros((5, 1))

    rectification_transform = None
    projection_matrix = None

    points = []
    map = None

    def append_coordinates(self, coordinates):
        self.points.append(coordinates.reshape(-1, 2))

