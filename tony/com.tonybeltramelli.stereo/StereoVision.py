__author__ = 'tbeltramelli'

from EpipolarGeometry import *
from DepthMap import *


class StereoVision:
    _result = None

    def __init__(self, left_video_path, right_video_path):
        left_img = UMedia.get_frame_from_video(left_video_path, 15)
        right_img = UMedia.get_frame_from_video(right_video_path, 15)

        #eg = EpipolarGeometry(self.get_stereo_images(left_img, right_img))
        dp = DepthMap(left_img, right_img)

    def get_stereo_images(self, raw_left, raw_right):
        left = cv2.pyrDown(raw_left)
        right = cv2.pyrDown(raw_right)

        return UMedia.combine_images(left, right)