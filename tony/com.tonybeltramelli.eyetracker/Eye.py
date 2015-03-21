__author__ = 'tbeltramelli'

from scipy.cluster.vq import *

from UMedia import *
from Filtering import *
from RegionProps import *
from UMath import *
from UGraphics import *
import operator

class Eye:
    _result = None
    _right_template = None
    _left_template = None

    def __init__(self, right_corner_path, left_corner_path):
        self._right_template = Filtering.apply_box_filter(Filtering.get_gray_scale_image(UMedia.get_image(right_corner_path)), 5)
        self._left_template = Filtering.apply_box_filter(Filtering.get_gray_scale_image(UMedia.get_image(left_corner_path)), 5)

    def process(self, img):
        self._result = img

        img = Filtering.apply_box_filter(Filtering.get_gray_scale_image(img), 5)

        pupil_position, pupil_radius = self.get_pupil(img, 40)
        iris_radius = self.get_iris(img, pupil_position, pupil_radius)
        glints_position = self.get_glints(img, 180, pupil_position, iris_radius)
        corners_position = self.get_eye_corners(img)

        UMedia.show(self._result)

    def get_pupil(self, img, threshold):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        height, width = img.shape
        side = (width * height) / 8

        st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, st, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, st, iterations=1)

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        props = RegionProps()

        radius = 0.0

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

                        if UMath.is_in_area(center[0], center[1], width, height):
                            if len(cnt) >= 5:
                                ellipse = cv2.fitEllipse(cnt)
                                cv2.ellipse(self._result, ellipse, (0, 0, 255), 1)

                            cv2.circle(self._result, center, int(radius), (0, 0, 255), 1)

                            return center, radius
        return (int(width / 2), int(height / 2)), radius

    def get_glints(self, img, threshold, pupil_position, iris_radius):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
        height, width = img.shape

        max_dist = iris_radius if iris_radius > 0 else (width + height) / 16

        c, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        props = RegionProps()

        coordinates = []

        for cnt in contours:
            properties = props.CalcContourProperties(cnt, ['Area', 'Length', 'Centroid', 'Extend', 'ConvexHull'])

            if properties['Extend'] > 0 and properties['Area'] < 100:
                for i, centroid in enumerate(properties['Centroid']):
                    if i == 0:
                        center = int(centroid), int(properties['Centroid'][i+1])

                        distance = np.sqrt(np.power(pupil_position[0] - center[0], 2) + np.power(pupil_position[1] - center[1], 2))

                        if distance < max_dist:
                            coordinates.append(center)
                            cv2.circle(self._result, center, 2, (0, 255, 0), 3)

        return coordinates

    def get_eye_corners(self, img):
        right = self._match(img, self._right_template)
        left = self._match(img, self._left_template)

        return [right, left]

    def _match(self, img, template):
        matching = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        height, width = template.shape

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching)

        cv2.rectangle(self._result, (max_loc[0] - width/2, max_loc[1] - height/2), (max_loc[0] + width/2, max_loc[1] + height/2), 2)

        return max_loc[0], max_loc[1]

    def get_iris(self, img, pupil_position, pupil_radius, angle_tolerance=3, min_magnitude=15, max_magnitude=20):
        if pupil_radius <= 1.0:
            return 0.0

        orientation, magnitude = self._get_gradient(img)

        max_iris_radius = pupil_radius * 5

        pupil_samples = UMath.get_circle_samples(pupil_position, pupil_radius)
        iris_samples = UMath.get_circle_samples(pupil_position, max_iris_radius)

        iris_radius_vote = dict()

        for sample in range(len(pupil_samples)):
            pupil_sample = (int(pupil_samples[sample][0]), int(pupil_samples[sample][1]))
            iris_sample = (int(iris_samples[sample][0]), int(iris_samples[sample][1]))

            normal = UMath.get_line_coordinates(pupil_sample, iris_sample)
            normal_angle = cv2.fastAtan2(pupil_sample[1] - pupil_position[1], pupil_sample[0] - pupil_position[0])

            for point in normal:
                i = point[1] - 1
                j = point[0] - 1

                if (i >= 0 and j >= 0) and (len(magnitude) > i and len(magnitude[i]) > j):
                    mag = magnitude[i][j]

                if min_magnitude < mag < max_magnitude:
                    angle = normal_angle + orientation[i][j] - 90
                    angle = angle - 360 if angle > 360 else angle

                    if angle < angle_tolerance:
                        radius = np.sqrt(np.power(point[0] - pupil_position[0], 2) + np.power(point[1] - pupil_position[1], 2))
                        radius = int(radius)

                        if radius not in iris_radius_vote:
                            iris_radius_vote[radius] = 0

                        iris_radius_vote[radius] += 1

            cv2.line(self._result, pupil_sample, iris_sample, UGraphics.hex_color_to_bgr(0xf2f378), 1)

        iris_radius = max(iris_radius_vote.iteritems(), key=operator.itemgetter(1))[0] if len(iris_radius_vote) > 0 else 0

        cv2.circle(self._result, pupil_position, iris_radius, (255, 0, 0), 1)

        return iris_radius

    def _get_gradient(self, img, granularity=10):
        height, width = img.shape

        sobel_horizontal = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobel_vertical = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        orientation = np.empty(img.shape)
        magnitude = np.empty(img.shape)

        for y in range(height):
            for x in range(width):
                orientation[y][x] = cv2.fastAtan2(sobel_horizontal[y][x], sobel_vertical[y][x])
                magnitude[y][x] = np.sqrt(np.power(sobel_horizontal[y][x], 2) + np.power(sobel_vertical[y][x], 2))

                if (x % granularity == 0) and (y % granularity == 0):
                    UGraphics.draw_vector(self._result, x, y, magnitude[y][x] / granularity, orientation[y][x])

        return orientation, magnitude
