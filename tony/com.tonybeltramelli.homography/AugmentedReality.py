__author__ = 'tbeltramelli'

from UMedia import *
from CameraCalibrator import *
from Filtering import *


class AugmentedReality:
    _video_path = None
    _camera_calibrator = None

    def __init__(self, video_path, output_path):
        self._video_path = video_path
        self._camera_calibrator = CameraCalibrator(output_path)

        UMedia.load_media(video_path, self.process)

    def process(self, img):
        img = cv2.pyrDown(img)
        img = Filtering.get_gray_scale_image(img)

        if not self._camera_calibrator.is_calibrated:
            self._camera_calibrator.calibrate(img)
        else:
            UMedia.show(img, self._camera_calibrator.get_undistorted_image(img))
