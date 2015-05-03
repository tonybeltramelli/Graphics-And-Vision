__author__ = 'tbeltramelli'

from StereoVision import *

#st = StereoVision("../data/media/", "../data/output/")
#st.epipolar_geometry("cameraLeft.mov", "cameraRight.mov", 11)
#st.stereo_vision_from_images([25, 37, 74, 80, 113])
#st.stereo_vision_from_video("cameraLeft2.mov", "cameraRight2.mov")

class ViSa:
    i = 0

    def process(self, images):
        left_img = images[0]
        right_img = images[1]

        self.i += 1

        cv2.imwrite("../data/output/l"+str(self.i)+".jpg", left_img)
        cv2.imwrite("../data/output/r"+str(self.i)+".jpg", right_img)

    def load(self):
        UMedia.load_videos(["../data/media/" + "cameraLeft2.mov", "../data/media/" + "cameraRight2.mov"], self.process)

s = ViSa()
s.load()
