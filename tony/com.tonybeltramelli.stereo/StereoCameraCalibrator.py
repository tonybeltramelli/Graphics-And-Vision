from UInteractive import *
from UGraphics import *


class StereoCameraCalibrator:
    is_calibrated = False

    _n = None
    _pattern_size = None
    _pattern_points = None

    _object_points = None
    left_points = None
    right_points = None
    _size = None

    left_camera_matrix = None
    right_camera_matrix = None
    left_distortion_coefficient = None
    right_distortion_coefficient = None
    rotation_matrix = None
    translation_vector = None
    fundamental_matrix = None

    left_rectification_transform = None
    right_rectification_transform = None
    left_projection_matrix = None
    right_projection_matrix = None
    disparity_to_depth_matrix = None

    left_map = None
    right_map = None

    def __init__(self, n=5, pattern_size=(9, 6)):
        self._n = n
        self._pattern_size = pattern_size

        self.is_calibrated = False

        self._pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        self._pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

        self._object_points = []
        self.left_points = []
        self.right_points = []

    def calibrate(self, left_img, right_img, to_draw=False):
        height, width, layers = left_img.shape

        self._size = (width, height)

        left_is_found, left_coordinates = cv2.findChessboardCorners(left_img, self._pattern_size, cv2.CALIB_CB_FAST_CHECK)
        right_is_found, right_coordinates = cv2.findChessboardCorners(right_img, self._pattern_size, cv2.CALIB_CB_FAST_CHECK)

        is_detected = left_is_found and right_is_found

        if is_detected and (self._n > 0):
            if to_draw:
                cv2.drawChessboardCorners(left_img, self._pattern_size, left_coordinates, left_is_found)
                cv2.drawChessboardCorners(right_img, self._pattern_size, right_coordinates, right_is_found)

                UMedia.show(UGraphics.get_combined_image(left_img, right_img))
                UInteractive.pause()

            self.left_points.append(left_coordinates.reshape(-1, 2))
            self.right_points.append(right_coordinates.reshape(-1, 2))
            self._object_points.append(self._pattern_points)

            self.stereo_calibrate()
            #self.stereo_rectify()

            self._n -= 1

            if self._n == 0:
                self.is_calibrated = True

        return is_detected

    def stereo_calibrate(self):
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

        calibrate = cv2.stereoCalibrate(self._object_points,
                                        self.left_points,
                                        self.right_points,
                                        self.left_camera_matrix,
                                        self.left_distortion_coefficient,
                                        self.right_camera_matrix,
                                        self.right_distortion_coefficient,
                                        self._size,
                                        criteria=criteria, flags=flags)

        reprojection_error_value = calibrate[0]
        self.left_camera_matrix = calibrate[1]
        self.left_distortion_coefficient = calibrate[2]
        self.right_camera_matrix = calibrate[3]
        self.right_distortion_coefficient = calibrate[4]
        self.rotation_matrix = calibrate[5]
        self.translation_vector = calibrate[6]
        essential_matrix = calibrate[7]
        self.fundamental_matrix = calibrate[8]

    def stereo_rectify(self):
        rectify = cv2.stereoRectify(self.left_camera_matrix,
                                    self.left_distortion_coefficient,
                                    self.right_camera_matrix,
                                    self.right_distortion_coefficient,
                                    self._size,
                                    self.rotation_matrix, self.translation_vector, alpha=0)

        self.left_rectification_transform = rectify[0]
        self.right_rectification_transform = rectify[1]
        self.left_projection_matrix = rectify[2]
        self.right_projection_matrix = rectify[3]
        self.disparity_to_depth_matrix = rectify[4]

    def get_undistorted_rectified_images(self, left_img, right_img):
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(self.left_camera_matrix,
                                                             self.left_distortion_coefficient,
                                                             self.left_rectification_transform,
                                                             self.left_projection_matrix,
                                                             self._size, cv2.CV_16SC2)

        right_map_x, right_map_y = cv2.initUndistortRectifyMap(self.right_camera_matrix,
                                                               self.right_distortion_coefficient,
                                                               self.right_rectification_transform,
                                                               self.right_projection_matrix,
                                                               self._size, cv2.CV_16SC2)

        left_img = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_img = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

        return left_img, right_img


