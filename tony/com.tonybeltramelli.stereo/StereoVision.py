__author__ = 'tbeltramelli'

from UMedia import *


class StereoVision:
    def __init__(self, left_video_path, right_video_path):
        UMedia.load_videos([left_video_path, right_video_path], self.process)

    def process(self, images):
        left = cv2.pyrDown(images[0])
        right = cv2.pyrDown(images[1])

        UMedia.show(UMedia.combine_images(left, right))

