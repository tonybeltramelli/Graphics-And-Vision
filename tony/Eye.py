__author__ = 'tbeltramelli'

import cv2
from pylab import *
import numpy as np
from Utils import *
from Filtering import *
from RegionProps import *

class Eye:
    def __init__(self):
        print("here")

    def process(self, img):
        img = Filtering.apply_box_filter(Filtering.get_gray_scale_image(img), 5)

        Utils.show(img, self.get_pupil(img, 70))
        #Utils.show(img, self.get_glints(img, 180))

    def get_pupil(self, img, threshold):
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        width, height = img.shape

        #img = img[height/4:3 * (height/4), width/4:3 * (width/4)]

        st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, st, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, st, iterations=1)

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        props = RegionProps()

        for cnt in contours:
            vals = props.CalcContourProperties(cnt, ['Area', 'Length', 'Centroid', 'Extend', 'ConvexHull'])

            perimeter = cv2.arcLength(cnt, True)
            radius = np.sqrt(vals['Area'] / np.pi)
            radius = 1.0 if radius == 0.0 else radius

            circularity = perimeter / (radius * 2 * np.pi)

            if ((circularity >= 0.0) and (circularity <= 1.5)) and ((vals['Area'] > 900) and (vals['Area'] < 3500)):
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(result, ellipse, (0, 255, 0), 2)

                for i, centroid in enumerate(vals['Centroid']):
                    if i == 0:
                        tuple = int(centroid), int(vals['Centroid'][i+1])
                        cv2.circle(result, tuple, int(radius), (0, 0, 255), 2)

        return result

    def get_glints(self, img, threshold):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
        width, height = img.shape

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        props = RegionProps()

        for cnt in contours:
            vals = props.CalcContourProperties(cnt, ['Area', 'Length', 'Centroid', 'Extend', 'ConvexHull'])

            print(vals['Extend'])

            if (vals['Area'] < 100):
                for i, centroid in enumerate(vals['Centroid']):
                    if i == 0:
                        tuple = int(centroid), int(vals['Centroid'][i+1])

                        if ((tuple[0] > width/4) and (tuple[0] < 3 * (width/4))) and ((tuple[0] > height/4) and (tuple[0] < 3 * (height/4))):
                            cv2.circle(result, tuple, 10, (0, 0, 255), 2)
                            cv2.circle(result, tuple, 2, (0, 255, 0), 3)

        return result
