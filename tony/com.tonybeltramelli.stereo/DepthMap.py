__author__ = 'tbeltramelli'

from UInteractive import *
from UMath import *


class DepthMap:
    _output_path = None

    _left_img = None
    _right_img = None

    _min_disparity = 0
    _block_size = 1

    _disparity_map = None

    def __init__(self, output_path):
        self._output_path = output_path

    def compute(self, left_img, right_img):
        self._left_img = left_img
        self._right_img = right_img

        self.show_setting_window()
        self.update()

        UInteractive.pause()
        return self._disparity_map

    def update(self):
        self._disparity_map = self.get_disparity_map(self._min_disparity, self._block_size)

        UMedia.show(self._disparity_map)

    def get_disparity_map(self, min_disparity=0, block_size=1):
        sgbm = cv2.StereoSGBM_create(min_disparity, min_disparity + 16, block_size,
                                     P1=8*3*block_size**2, P2=32*3*block_size**2,
                                     disp12MaxDiff=1, preFilterCap=63,
                                     uniquenessRatio=10, speckleWindowSize=100,
                                     speckleRange=32, mode=cv2.STEREO_SGBM_MODE_HH)

        disparity = sgbm.compute(self._left_img, self._right_img)

        min_val = np.min(np.min(disparity, axis=1), axis=0)
        max_val = np.max(np.max(disparity, axis=1), axis=0)

        height, width = disparity.shape
        result = np.zeros((height, width), np.uint8)

        for y in range(height):
            for x in range(width):
                value = UMath.normalize(0, 255, disparity[y][x], min_val, max_val)
                result[y][x] = value

        return result

    def save_point_cloud(self, disparity, disparity_to_depth_matrix):
        points = cv2.reprojectImageTo3D(disparity, disparity_to_depth_matrix)

        mask = disparity > disparity.min()
        points = points[mask].reshape(-1, 3)

        points = map(self.get_color, points)

        with open(self._output_path + "/point_cloud.ply", "w") as filename:
            ply_header = '''ply
format ascii 1.0
element vertex %(num)d
property float x
property float y
property float z
property int r
property int g
property int b
end_header
'''
            filename.write(ply_header % dict(num=len(points)))
            np.savetxt(filename, points, "%f %f %f %i %i %i", newline="\n")

    def get_color(self, p):
        r = 100
        g = 100
        b = 100

        return [p[0], p[1], p[2], r, g, b]

    def show_setting_window(self):
        cv2.namedWindow("DepthMap", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("min disparity", "DepthMap", 1, 32, self.update_min_disparity)
        cv2.createTrackbar("block size", "DepthMap", 1, 5, self.update_block_size)

    def update_min_disparity(self, value):
        if value % 16 == 0:
            self._min_disparity = value
            self.update()

    def update_block_size(self, value):
        if value % 2 != 0:
            self._block_size = value
            self.update()