import cv2
from pylab import *

class CameraCalibrator:
    is_calibrated = False

    _n = 0
    _output_path = None

    _camera_matrix = None
    _distortion_coefficient = None
    _rotation_vectors = None
    _translation_vectors = None

    _pattern_size = None
    _square_size = None
    _pattern_points = None
    _obj_points = None
    _img_points = None

    def __init__(self, output_path, n=5, pattern_size=(9, 6), square_size=2.0):
        self._output_path = output_path
        self._n = n
        self._pattern_size = pattern_size
        self._square_size = square_size

        self.is_calibrated = False

        self._pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        self._pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        self._pattern_points *= square_size

        self._camera_matrix = np.zeros((3, 3))
        self._distortion_coefficient = np.zeros(4)
        self._rotation_vectors = np.zeros((3, 3))
        self._translation_vectors = np.zeros((3, 3))

        self._obj_points = []
        self._img_points = []

    def calibrate(self, img):
        height, width = img.shape
        is_found, coordinates = cv2.findChessboardCorners(img, self._pattern_size)

        if is_found & (self._n > 0):
            corners_first = []

            for p in coordinates:
                corners_first.append(p[0])

            img_points_first = np.asarray(corners_first, np.float64)

            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, coordinates, (5, 5), (-1, -1), term)

            self._img_points.append(coordinates.reshape(-1, 2))
            self._obj_points.append(self._pattern_points)

            self._n -= 1
            if self._n == 0:
                rms, camera_matrix, distortion_coefficient, rotation_vectors, translation_vectors = cv2.calibrateCamera(self._obj_points, self._img_points, (width, height), self._camera_matrix, self._distortion_coefficient, flags=0)
                self.is_calibrated = True

                np.save(self._output_path + "/camera_matrix", camera_matrix)
                np.save(self._output_path + "/distortion_coefficient", distortion_coefficient)
                np.save(self._output_path + "/rotation_vectors", rotation_vectors)
                np.save(self._output_path + "/translation_vectors", translation_vectors)
                np.save(self._output_path + "/chessSquare_size", self._square_size)
                np.save(self._output_path + "/img_points", self._img_points)
                np.save(self._output_path + "/obj_points", self._obj_points)
                np.save(self._output_path + "/img_points_first", img_points_first)

    def get_undistorted_image(self, img):
        img = cv2.undistort(img, self._camera_matrix, self._distortion_coefficient)
        return img
