__author__ = 'tbeltramelli'

from StereoVision import *

st = StereoVision("../data/media/", "../data/output/")
st.epipolar_geometry("cameraLeft.mov", "cameraRight.mov", 11)
st.stereo_vision_from_images([25, 37, 74, 80, 113], [172, 252])

st = StereoVision("../data/media/", "../data/output/")
st.stereo_vision_from_video("cameraLeft2.mov", "cameraRight2.mov")

