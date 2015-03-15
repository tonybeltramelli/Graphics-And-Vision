__author__ = 'tbeltramelli'

from scipy.cluster.vq import *

from UMedia import *
from Filtering import *
from RegionProps import *
from UMath import *
from UGraphics import *

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

        pupil = self.get_pupil(img, 40)
        glints = self.get_glints(img, 180, pupil)
        #corners = self.get_eye_corners(img)
        iris = self.get_iris(img)

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

                            return center
        return int(width / 2), int(height / 2)


    def _detect_pupil_k_means(self, img, intensity_weight=2, side=100, clusters=5):
        img = cv2.resize(img, (side, side))

        height, width = img.shape
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

    def get_glints(self, img, threshold, pupil_position):
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
        height, width = img.shape

        max_dist = ((width + height) / 2) / 8

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

    def get_iris(self, img):
        self._draw_gradient_image(img)

        return [0, 0]

    def _draw_gradient_image(self, img, granularity=10):
        height, width = img.shape

        sobel_horizontal = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobel_vertical = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        for y in range(height):
            for x in range(width):
                if (x % granularity == 0) and (y % granularity == 0):
                    orientation = cv2.fastAtan2(sobel_horizontal[y][x], sobel_vertical[y][x])
                    magnitude = np.sqrt((sobel_horizontal[y][x] * sobel_horizontal[y][x]) + (sobel_vertical[y][x] * sobel_vertical[y][x]))
                    
                    UGraphics.draw_vector(self._result, x, y, magnitude / granularity, orientation)

    def FindEllipseContour (self, img, gradient_magnitude, estimated_center, estimated_radius):
        point_number = 30
        points = self.get_circle_samples(estimated_center, estimated_radius)

        t = 0

        pupil = np.zeros((point_number, 1, 2)).astype(np.float32)

        #for (x, y, dx , dy) in points:

    def get_circle_samples(self, center=(0, 0), radius=1, point_number=30):
        s = np.linspace(0, 2 * math.pi, point_number)
        return [(radius * np.cos(t) + center[0], radius * np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s]
