__author__ = 'tbeltramelli'

#from StereoVision import *

#st = StereoVision("../data/media/cameraLeft.mov", "../data/media/cameraRight.mov", "../data/output")

from StereoCameraCalibrator import *

left_img1 = cv2.pyrDown(UMedia.get_frame_from_video("../data/media/cameraLeft.mov", 20))
right_img1 = cv2.pyrDown(UMedia.get_frame_from_video("../data/media/cameraRight.mov", 20))

left_img2 = cv2.pyrDown(UMedia.get_frame_from_video("../data/media/cameraLeft.mov", 50))
right_img2 = cv2.pyrDown(UMedia.get_frame_from_video("../data/media/cameraRight.mov", 50))

#UMedia.show(left_img1, right_img1, left_img2, right_img2)
#UInteractive.pause("pause")

calibrator = StereoCameraCalibrator(20)
calibrator.calibrate(left_img1, right_img1)
calibrator.calibrate(left_img2, right_img2)
