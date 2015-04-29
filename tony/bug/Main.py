__author__ = 'tbeltramelli'

from StereoCameraCalibrator import *

calibrator = StereoCameraCalibrator()

a = [25, 37, 74, 80, 113]

for index in a:
    l = cv2.imread("img/l" + str(index) + ".jpg")
    r = cv2.imread("img/r" + str(index) + ".jpg")

    calibrator.calibrate(l, r, True)
