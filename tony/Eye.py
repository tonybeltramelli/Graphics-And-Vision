__author__ = 'tbeltramelli'

import cv2
from pylab import *
import numpy as np
from Utils import *
from Filtering import *
from RegionProps import *
from scipy.cluster.vq import *


class Eye:
    _result = None
    _right_template = None
    _left_template = None

    def __init__(self, right_corner_path, left_corner_path):
        self._right_template = Filtering.apply_box_filter(Filtering.get_gray_scale_image(Utils.get_image(right_corner_path)), 5)
        self._left_template = Filtering.apply_box_filter(Filtering.get_gray_scale_image(Utils.get_image(left_corner_path)), 5)

    def process(self, img):
        self._result = img

        img = Filtering.apply_box_filter(Filtering.get_gray_scale_image(img), 5)

        #pupils = self.get_pupil(img, 40)
        #glints = self.get_glints(img, 180)
        #corners = self.get_eye_corners(img)
        iris = self.get_iris(img)

        Utils.show(self._result)

    def get_pupil(self, img, threshold):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        width, height = img.shape
        side = (width * height) / 8

        st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, st, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, st, iterations=1)

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        props = RegionProps()

        coordinates = []

        for cnt in contours:
            properties = props.CalcContourProperties(cnt, ['Area', 'Length', 'Centroid', 'Extend', 'ConvexHull'])

            perimeter = cv2.arcLength(cnt, True)
            radius = np.sqrt(properties['Area'] / np.pi)
            radius = 1.0 if radius == 0.0 else radius

            circularity = perimeter / (radius * 2 * np.pi)

            if ((circularity >= 0.0) and (circularity <= 1.5)) and ((properties['Area'] > 900) and (properties['Area'] < side)):
                for i, centroid in enumerate(properties['Centroid']):
                    if i == 0:
                        center = int(centroid), int(properties['Centroid'][i+1])

                        if Utils.is_in_area_center(center[0], center[1], width, height):
                            if len(cnt) >= 5:
                                ellipse = cv2.fitEllipse(cnt)
                                cv2.ellipse(self._result, ellipse, (0, 0, 255), 1)

                            coordinates.append(center)
                            cv2.circle(self._result, center, int(radius), (0, 0, 255), 1)

        return coordinates

    def _detect_pupil_k_means(self, img, intensity_weight=2, side=100, clusters=5):
        img = cv2.resize(img, (side, side))

        width, height = img.shape
        rows, columns = np.meshgrid(range(width), range(height))

        x = rows.flatten()
        y = columns.flatten()
        intensity = img.flatten()

        features = np.zeros((len(x), 3))
        features[:, 0] = intensity * intensity_weight
        features[:, 1] = y
        features[:, 2] = x

        features = np.array(features, 'f')

        centroids, variance = kmeans(features, clusters)
        label, distance = vq(features, centroids)

        labels = np.array(np.reshape(label, (width, height)))

        f = figure(1)
        imshow(labels)
        f.canvas.draw()
        f.show()

    def get_glints(self, img, threshold):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
        width, height = img.shape

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        props = RegionProps()

        coordinates = []

        for cnt in contours:
            properties = props.CalcContourProperties(cnt, ['Area', 'Length', 'Centroid', 'Extend', 'ConvexHull'])

            if properties['Extend'] > 0 and properties['Area'] < 100:
                for i, centroid in enumerate(properties['Centroid']):
                    if i == 0:
                        center = int(centroid), int(properties['Centroid'][i+1])

                        if Utils.is_in_area_center(center[0], center[1], width, height):
                            coordinates.append(center)
                            cv2.circle(self._result, center, 2, (0, 255, 0), 3)

        return coordinates

    def get_eye_corners(self, img):
        right = self._match(img, self._right_template)
        left = self._match(img, self._left_template)

        return [right, left]

    def _match(self, img, template):
        matching = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        width, height = template.shape

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching)

        cv2.rectangle(self._result, (max_loc[0] - height/2, max_loc[1] - width/2), (max_loc[0] + height/2, max_loc[1] + width/2), 2)

        return max_loc[0], max_loc[1]

    def get_iris(self, img):
        self.gradient_image_info(img)

        return [0, 0]

    def gradient_image_info(self, img):
        height, width = img.shape

        sobel_horizontal = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobel_vertical = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        angle = np.empty(img.shape)
        magnitude = np.empty(img.shape)

        for y in range(height):
            for x in range(width):
                if (x % 20 == 0) and (y % 20 == 0):
                    angle[y][x] = cv2.fastAtan2(sobel_horizontal[y][x], sobel_vertical[y][x])
                    magnitude[y][x] = np.sqrt((sobel_horizontal[y][x] * sobel_horizontal[y][x]) + (sobel_vertical[y][x] * sobel_vertical[y][x]))

                    #cv2.line(self._result, (x, y), (x + 10, y + 2), (0, 255, 0), 1)
