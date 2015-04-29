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

    _i = 0
    #25 37 74 80 94 101 113 130

    def __init__(self, left_video_path, right_video_path, output_path):
        #left_img = cv2.pyrDown(UMedia.get_frame_from_video(left_video_path, 15))
        #right_img = cv2.pyrDown(UMedia.get_frame_from_video(right_video_path, 15))
        #eg = EpipolarGeometry(UGraphics.get_combined_image(left_img, right_img))

        #self._calibrator = StereoCameraCalibrator(output_path)
        #self._depth = DepthMap(output_path)

        #UMedia.load_videos([left_video_path, right_video_path], self.process)

        self.see(left_video_path, right_video_path, output_path, 25)
        self.see(left_video_path, right_video_path, output_path, 37)
        self.see(left_video_path, right_video_path, output_path, 74)
        self.see(left_video_path, right_video_path, output_path, 80)
        self.see(left_video_path, right_video_path, output_path, 94)
        self.see(left_video_path, right_video_path, output_path, 101)
        self.see(left_video_path, right_video_path, output_path, 113)
        self.see(left_video_path, right_video_path, output_path, 130)


    def see(self, left_video_path, right_video_path, output_path, i):
        l1 = UMedia.get_frame_from_video(left_video_path, i)
        r1 = UMedia.get_frame_from_video(right_video_path, i)
        UMedia.show(UGraphics.get_combined_image(cv2.pyrDown(l1), cv2.pyrDown(r1)))
        UInteractive.pause(str(i))

        cv2.imwrite(output_path + "/l" + str(i) + ".jpg", l1)
        cv2.imwrite(output_path + "/r" + str(i) + ".jpg", r1)

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