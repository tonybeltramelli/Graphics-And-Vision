__author__ = 'tbeltramelli'

from UInteractive import *


class EpipolarGeometry:
    _points = []
    _img = None
    _raw_img = None
    _fundamental_matrix = None
    _epipole = None

    _MAX_POINT_NUMBER = 16

    is_ready = False

    def __init__(self, img, define_manually=True):
        self._img = img
        self._raw_img = copy(self._img)

        if define_manually:
            left_points, right_points = self.get_manually_selected_features()

            fundamental_matrix, mask = cv2.findFundamentalMat(left_points, right_points)

            self.show_lines(left_points, right_points, fundamental_matrix)

    def show_lines(self, left_points, right_points, fundamental_matrix):
        self.build_epipolar_lines(left_points, fundamental_matrix, False)
        self.build_epipolar_lines(right_points, fundamental_matrix, True)

        UMedia.show(self._raw_img)
        UInteractive.pause()

    def get_manually_selected_features(self):
        UMedia.show(self._img)
        cv2.setMouseCallback("image 0", self.mouse_event)

        UInteractive.pause("Select 8 points in each image")

        left_points = np.array(self._points[::2])
        right_points = np.array(self._points[1::2])

        return left_points, right_points

    def build_epipolar_lines(self, points, fundamental_matrix, is_right, show_lines=True):
        lines = cv2.computeCorrespondEpilines(points, 2 if is_right else 1, fundamental_matrix)
        lines = lines.reshape(-1, 3)

        if show_lines:
            self.draw_lines(self._raw_img, lines, points, is_right)

    def draw_lines(self, img, lines, points, is_right):
        height, width, layers = img.shape

        color = (0, 0, 255) if not is_right else (255, 0, 0)
        x_gap_point = 0 if not is_right else width / 2
        x_gap_line = 0 if is_right else width / 2

        for height, row in zip(lines, points):
            x_start, y_start = map(int, [0, -height[2]/height[1]])
            x_end, y_end = map(int, [width/2, -(height[2]+height[0]*(width/2))/height[1]])
            row = map(int, row)

            cv2.line(img, (x_start + x_gap_line, y_start), (x_end + x_gap_line, y_end), color, 1)
            cv2.circle(img, (row[0] + x_gap_point, row[1]), 3, color)

        return img

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
                print "done"
                self.is_ready = True

            UMedia.show(self._img)
        elif event == cv2.EVENT_RBUTTONUP:
            self._points = []
            self._img = copy(self._raw_img)

            UMedia.show(self._img)