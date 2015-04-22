__author__ = 'tbeltramelli'

from UMedia import *
from FundamentalMatrix import *


class StereoVision:
    _result = None
    _fundamental_matrix = None


    def __init__(self, left_video_path, right_video_path):
       UMedia.load_videos([left_video_path, right_video_path], self.process)

    def process(self, images):
        left = cv2.pyrDown(images[0])
        right = cv2.pyrDown(images[1])

        self._result = UMedia.combine_images(left, right)

        if self._fundamental_matrix is None:
            self._fundamental_matrix = FundamentalMatrix(self._result)




