__author__ = 'tbeltramelli'

import cv2
from UMedia import *

class FundamentalMatrix:
    _points = []
    _img = None
    _raw_img = None

    _MAX_POINT_NUMBER = 16

    is_ready = False

    def __init__(self, img):
        self._img = img
        self._raw_img = copy(img)

        UMedia.show(self._img)
        cv2.setMouseCallback("image 0", self.mouse_event)

        cv2.waitKey(0)

    def mouse_event(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_ready:
                return

            height, width, layers = self._img.shape

            point = (x, y)

            color = (0, 0, 255) if len(self._points) % 2 == 0 else (255, 0, 0)

            cv2.circle(self._img, point, 3, color, thickness=-1)

            point = (point[0] - (0 if len(self._points) % 2 == 0 else width / 2), point[1])

            point = (point[0], point[1], 1)
            self._points.append(point)

            if len(self._points) is self._MAX_POINT_NUMBER:
                self.is_ready = True

            UMedia.show(self._img)
        elif event == cv2.EVENT_RBUTTONUP:
            self._points = []
            self._img = copy(self._raw_img)

            UMedia.show(self._img)