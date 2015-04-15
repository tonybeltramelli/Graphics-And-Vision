__author__ = 'tbeltramelli'

from UMedia import *
from CameraCalibrator import *
from Filtering import *
from Camera import *
from UMath import *
from UGraphics import *
from AHomography import *


class AugmentedReality:
    _video_path = None
    _pattern = None
    _camera_calibrator = None
    _result = None

    def __init__(self, video_path, output_path, pattern_path):
        self._video_path = video_path
        self._camera_calibrator = CameraCalibrator(output_path)

        self._pattern = UMedia.get_image(pattern_path)

        UMedia.load_media(video_path, self.process)

    def process(self, img):
        img = cv2.pyrDown(img)
        self._result = img
        img = Filtering.get_gray_scale_image(img)

        if not self._camera_calibrator.is_calibrated:
            self._camera_calibrator.calibrate(img)
        else:
            #UMedia.show(img, self._camera_calibrator.get_undistorted_image(img))
            self.augment(img)

    def get_cube_points(self, center, size):
        points = []

        #bottom
        points.append([center[0]-size, center[1]-size, center[2]-2*size])#(0)5
        points.append([center[0]-size, center[1]+size, center[2]-2*size])#(1)7
        points.append([center[0]+size, center[1]+size, center[2]-2*size])#(2)8
        points.append([center[0]+size, center[1]-size, center[2]-2*size])#(3)6
        points.append([center[0]-size, center[1]-size, center[2]-2*size]) #same as first to close plot

        #top
        points.append([center[0]-size, center[1]-size, center[2]])#(5)1
        points.append([center[0]-size, center[1]+size, center[2]])#(6)3
        points.append([center[0]+size, center[1]+size, center[2]])#(7)4
        points.append([center[0]+size, center[1]-size, center[2]])#(8)2
        points.append([center[0]-size, center[1]-size, center[2]]) #same as first to close plot

        #vertical sides
        points.append([center[0]-size, center[1]-size, center[2]])
        points.append([center[0]-size, center[1]+size, center[2]])
        points.append([center[0]-size, center[1]+size, center[2]-2*size])
        points.append([center[0]+size, center[1]+size, center[2]-2*size])
        points.append([center[0]+size, center[1]+size, center[2]])
        points.append([center[0]+size, center[1]-size, center[2]])
        points.append([center[0]+size, center[1]-size, center[2]-2*size])

        return np.array(points).T

    def augment(self, img, pattern_size=(9, 6)):
        camera_matrix, distortion_coefficient = self._camera_calibrator.load_calibration()

        calibration_pattern = cv2.pyrDown(self._pattern)
        calibration_pattern = Filtering.get_gray_scale_image(calibration_pattern)

        is_found, pattern_corners = cv2.findChessboardCorners(calibration_pattern, pattern_size)
        is_found, img_corners = cv2.findChessboardCorners(img, pattern_size)

        if not is_found:
            return

        image_points_position = []
        pattern_points_position = []

        pos = np.array([0, pattern_size[0] - 1, pattern_size[0] * (pattern_size[1] - 1), (pattern_size[0] * pattern_size[1]) - 1])

        for i in pos:
            image_points_position.append(img_corners[i][0])
            pattern_points_position.append(pattern_corners[i][0])

        image_points_position = np.array(image_points_position)
        pattern_points_position = np.array(pattern_points_position)

        homography = AHomography.estimate_homography(pattern_points_position, image_points_position)

        camera = Camera(hstack((camera_matrix, dot(camera_matrix, np.array([[0], [0], [-1]])))))
        camera = Camera(dot(homography, camera.P))

        calibration_inverse = np.linalg.inv(camera_matrix)
        rotation_translation_matrix = UMath.get_rotation_translation_matrix(dot(calibration_inverse, camera.P[:, :3]))

        camera.P = dot(camera_matrix, rotation_translation_matrix)

        cube = self.get_cube_points([0, 0, 0], 0.5)
        coordinates = camera.project(UMath.to_homogenious(cube))

        for i in range(0, len(cube[0])):
            start_point = (int(coordinates[0, i - 1]), int(coordinates[1, i - 1]))
            end_point = (int(coordinates[0, i]), int(coordinates[1, i]))

            cv2.line(self._result, start_point, end_point, UGraphics.hex_color_to_bgr(0xf2f378), 1)

        UMedia.show(self._result)
