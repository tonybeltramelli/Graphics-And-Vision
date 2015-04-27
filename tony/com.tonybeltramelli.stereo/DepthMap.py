__author__ = 'tbeltramelli'

from UInteractive import *
from UMath import *

class DepthMap:
    _left_img = None
    _right_img = None

    def __init__(self, left_img, right_img):
        self._left_img = cv2.pyrDown(left_img)
        self._right_img = cv2.pyrDown(right_img)

        map = self.get_disparity_map(32, 3)

        UMedia.show(map)
        UInteractive.pause("pause")

    def get_disparity_map(self, min_disparity=0, block_size=1):
        sgbm = cv2.StereoSGBM_create(min_disparity, min_disparity + 16, block_size,
                                     P1=8*3*block_size**2, P2=32*3*block_size**2,
                                     disp12MaxDiff=1, preFilterCap=63,
                                     uniquenessRatio=10, speckleWindowSize=100,
                                     speckleRange=32, mode=cv2.STEREO_SGBM_MODE_HH)

        disparity = sgbm.compute(self._left_img, self._right_img)

        min = np.min(np.min(disparity, axis=1), axis=0)
        max = np.max(np.max(disparity, axis=1), axis=0)

        height, width = disparity.shape
        result = np.zeros((height, width), np.uint8)

        for y in range(height):
            for x in range(width):
                value = UMath.normalize(0, 255, disparity[y][x], min, max)
                result[y][x] = value

        return result