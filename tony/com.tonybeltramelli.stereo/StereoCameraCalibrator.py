import cv2
from pylab import *
from UInteractive import *
from UGraphics import *

class StereoCameraCalibrator:
    is_calibrated = False

    _output_path = None
    _n = None
    _pattern_size = None
    _pattern_points = None

    _object_points = None
    _left_points = None
    _right_points = None
    _width = None
    _height = None

    left_camera_matrix = None
    right_camera_matrix = None
    left_distortion_coefficient = None
    right_distortion_coefficient = None
    rotation_matrix = None
    translation_vector = None
    essential_matrix = None
    fundamental_matrix = None

    left_rectification_transform = None
    right_rectification_transform = None
    left_projection_matrix = None
    right_projection_matrix = None
    disparity_to_depth_matrix = None

    left_map = None
    right_map = None

    def __init__(self, output_path, n=5, pattern_size=(9, 6), square_size=2.0):
        self._output_path = output_path
        self._n = n
        self._pattern_size = pattern_size

        self.is_calibrated = False

        self._pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        self._pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        self._pattern_points *= square_size

        self._object_points = []
        self._left_points = []
        self._right_points = []

    def calibrate(self, left_img, right_img, to_draw=False, to_override=False):
        self._height, self._width, layers = left_img.shape

        if not to_override:
            try:
                self.load_calibration()
            except IOError, e:
                self.compute_calibration(left_img, right_img, to_draw)
        else:
            self.compute_calibration(left_img, right_img, to_draw)

    def compute_calibration(self, left_img, right_img, to_draw):
        left_is_found, left_coordinates = cv2.findChessboardCorners(left_img, self._pattern_size, cv2.CALIB_CB_FAST_CHECK)
        right_is_found, right_coordinates = cv2.findChessboardCorners(right_img, self._pattern_size, cv2.CALIB_CB_FAST_CHECK)

        if (left_is_found and right_is_found) and (self._n > 0):
            if to_draw:
                cv2.drawChessboardCorners(left_img, self._pattern_size, left_coordinates, left_is_found)
                cv2.drawChessboardCorners(right_img, self._pattern_size, right_coordinates, right_is_found)

                UMedia.show(UGraphics.get_combined_image(left_img, right_img))

            self._left_points.append(left_coordinates)
            self._right_points.append(right_coordinates)
            self._object_points.append(self._pattern_points)

            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
            flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

            calibrate = cv2.stereoCalibrate(self._object_points,
                                            self._left_points,
                                            self._right_points,
                                            self.left_camera_matrix,
                                            self.left_distortion_coefficient,
                                            self.right_camera_matrix,
                                            self.right_distortion_coefficient,
                                            (self._width, self._height), criteria=criteria, flags=flags)

            reprojection_error_value = calibrate[0]
            self.left_camera_matrix = calibrate[1]
            self.left_distortion_coefficient = calibrate[2]
            self.right_camera_matrix = calibrate[3]
            self.right_distortion_coefficient = calibrate[4]
            self.rotation_matrix = calibrate[5]
            self.translation_vector = calibrate[6]
            self.essential_matrix = calibrate[7]
            self.fundamental_matrix = calibrate[8]

            self.rectify(self.rotation_matrix, self.translation_vector)
            self.compute_undistorted_rectified_map()

            self._n -= 1
            print self._n
            if self._n == 0:
                self.save_calibration()

    def rectify(self, rotation_matrix, translation_vector):
        rectify = cv2.stereoRectify(self.left_camera_matrix,
                                    self.left_distortion_coefficient,
                                    self.right_camera_matrix,
                                    self.right_distortion_coefficient,
                                    (self._width, self._height),
                                    rotation_matrix, translation_vector, alpha=0)

        self.left_rectification_transform = rectify[0]
        self.right_rectification_transform = rectify[1]
        self.left_projection_matrix = rectify[2]
        self.right_projection_matrix = rectify[3]
        self.disparity_to_depth_matrix = rectify[4]

    def compute_undistorted_rectified_map(self):
        self.left_map = cv2.initUndistortRectifyMap(self.left_camera_matrix,
                                                    self.left_distortion_coefficient,
                                                    self.left_rectification_transform,
                                                    self.left_projection_matrix,
                                                    (self._width, self._height), cv2.CV_16SC2)

        self.right_map = cv2.initUndistortRectifyMap(self.right_camera_matrix,
                                                     self.right_distortion_coefficient,
                                                     self.right_rectification_transform,
                                                     self.right_projection_matrix,
                                                     (self._width, self._height), cv2.CV_16SC2)

    def get_undistorted_rectified_images(self, left_img, right_img):
        left_img = cv2.remap(left_img, self.left_map[0], self.left_map[1], cv2.INTER_LINEAR)
        right_img = cv2.remap(right_img, self.right_map[0], self.right_map[1], cv2.INTER_LINEAR)

        return left_img, right_img

    def save_calibration(self):
        np.save(self._output_path + "/left_map", self.left_map)
        np.save(self._output_path + "/right_map", self.right_map)

        self.is_calibrated = True

    def load_calibration(self):
        self.left_map = np.load(self._output_path + "/left_map.npy")
        self.right_map = np.load(self._output_path + "/right_map.npy")

        self.is_calibrated = True


