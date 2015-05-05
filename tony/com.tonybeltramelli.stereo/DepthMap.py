__author__ = 'tbeltramelli'

from UInteractive import *
from UMath import *
from Filtering import *
from UGraphics import *

class DepthMap:
    _output_path = None

    _left_img = None
    _right_img = None

    _min_disparity = -16
    _block_size = 5

    _disparity_map = None

    def __init__(self, output_path):
        self._output_path = output_path

    def compute(self, left_img, right_img, to_resize=False):
        self._left_img = left_img if not to_resize else UGraphics.get_resized_image(left_img, 0.5)
        self._right_img = right_img if not to_resize else UGraphics.get_resized_image(right_img, 0.5)

        self.show_setting_window()
        self.update()

        UInteractive.pause()
        return self._disparity_map

    def update(self):
        self._disparity_map = self.get_disparity_map(self._min_disparity, self._block_size)

        UMedia.show(self._disparity_map)

    def get_disparity_map(self, min_disparity=0, block_size=1, to_equalize=False):
        print min_disparity, block_size

        sgbm = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=192, blockSize=block_size,
                                     P1=600, P2=2400,
                                     disp12MaxDiff=10, preFilterCap=4,
                                     uniquenessRatio=1, speckleWindowSize=150,
                                     speckleRange=2, mode=cv2.STEREO_SGBM_MODE_HH)

        left_img = Filtering.get_gray_scale_image(self._left_img)
        right_img = Filtering.get_gray_scale_image(self._right_img)

        if to_equalize:
            left_img = cv2.equalizeHist(left_img, left_img)
            right_img = cv2.equalizeHist(right_img, right_img)

        disparity = sgbm.compute(left_img, right_img)
        disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return disparity

    def save_point_cloud(self, disparity, disparity_to_depth_matrix=None):
        if disparity_to_depth_matrix is None:
            height, width, layers = self._left_img.shape
            disparity_to_depth_matrix = np.float32([[1, 0, 0, -0.5 * width],
                                                    [0, -1, 0, 0.5 * height],
                                                    [0, 0, 0, - 0.8 * width],
                                                    [0, 0, 1, 0]])

        points = cv2.reprojectImageTo3D(disparity, disparity_to_depth_matrix)
        colors = cv2.cvtColor(self._left_img, cv2.COLOR_BGR2RGB)

        mask = disparity > disparity.min()
        points = points[mask].reshape(-1, 3)
        colors = colors[mask].reshape(-1, 3)

        points = np.hstack([points, colors])

        with open(self._output_path + "/point_cloud.ply", "w") as filename:
            ply_header = '''ply
format ascii 1.0
element vertex %(num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
            filename.write(ply_header % dict(num=len(points)))
            np.savetxt(filename, points, "%f %f %f %d %d %d", newline="\n")

    def show_setting_window(self):
        cv2.namedWindow("DepthMap", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("min disparity", "DepthMap", 0, 128, self.update_min_disparity)
        cv2.createTrackbar("block size", "DepthMap", 5, 25, self.update_block_size)

    def update_min_disparity(self, value):
        if value % 16 == 0:
            self._min_disparity = value - 64
            self.update()

    def update_block_size(self, value):
        if value % 2 != 0:
            self._block_size = value
            self.update()