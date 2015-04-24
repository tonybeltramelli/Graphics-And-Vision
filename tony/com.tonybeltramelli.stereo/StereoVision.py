__author__ = 'tbeltramelli'

from FundamentalMatrix import *


class StereoVision:
    _result = None

    def __init__(self, left_video_path, right_video_path):
        self._result = self.get_stereo_images(UMedia.get_frame_from_video(left_video_path, 1), UMedia.get_frame_from_video(right_video_path, 1))

        fm = FundamentalMatrix()
        m = fm.get(self._result)

        #self.compute_epipole(matrix)

    def get_stereo_images(self, raw_left, raw_right):
        left = cv2.pyrDown(raw_left)
        right = cv2.pyrDown(raw_right)

        return UMedia.combine_images(left, right)

    def compute_epipole(self, F):
        U, S, V = linalg.svd(F)
        e = V[-1]
        return e/e[2]