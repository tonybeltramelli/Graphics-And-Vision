__author__ = 'tbeltramelli'

#from StereoVision import *

#st = StereoVision("../data/media/cameraLeft2.mov", "../data/media/cameraRight2.mov", "../data/output")

from StereoCameraCalibrator import *

calibrator = StereoCameraCalibrator()


def feed(index):
    l = UMedia.get_image("../data/output/l" + str(index) + ".jpg")
    r = UMedia.get_image("../data/output/r" + str(index) + ".jpg")

    calibrator.calibrate(l, r, True)

a = [25, 80, 113]

for i in a:
    feed(i)