from numpy.core.umath import right_shift

__author__ = 'tbeltramelli'

from EpipolarGeometry import *
from StereoCameraCalibrator import *
from DepthMap import *
from Filtering import *


class StereoVision:
    _result = None
    _calibrator = None
    _depth = None

    def __init__(self, left_video_path, right_video_path, output_path):
        #left_img = cv2.pyrDown(UMedia.get_frame_from_video(left_video_path, 15))
        #right_img = cv2.pyrDown(UMedia.get_frame_from_video(right_video_path, 15))
        #eg = EpipolarGeometry(UGraphics.get_combined_image(left_img, right_img))

        self._calibrator = StereoCameraCalibrator(output_path)
        #self._depth = DepthMap(output_path)

        #UMedia.load_videos([left_video_path, right_video_path], self.process)

    def process(self, images):
        #self._calibrator.calibrate(images[0], images[1], True)
        images = map(cv2.pyrDown, images)
        self._i += 1

        UMedia.show(UGraphics.get_combined_image(images[0], images[1]))

        print(self._i)

        UInteractive.pause()

        #images = map(cv2.pyrDown, images)

        #if self._calibrator.is_calibrated:
        #    left_img, right_img = self._calibrator.get_undistorted_rectified_images(images[0], images[1])
        #    #disparity_map = self._depth.get(left_img, right_img)

        #    original_img = UGraphics.get_combined_image(images[0], images[1], 0.5)
        #    rectified_img = UGraphics.get_combined_image(left_img, right_img, 0.5)

        #    UMedia.show(UGraphics.get_combined_image(original_img, rectified_img, use_horizontally=False))

            #self._depth.save_point_cloud(disparity_map, self._calibrator.disparity_to_depth_matrix)

        #else:
        #   self._calibrator.calibrate(images[0], images[1])