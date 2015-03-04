__author__ = 'tbeltramelli'

import cv2

from Eye import *
from Utils import *


class Tracker:
    def __init__(self, path):
        e = Eye("../Sequences/right_corner.jpg", "../Sequences/left_corner.jpg")
        Utils.load_video(path, e.process)
        #e.process(Utils.get_image("../Sequences/Eye1.jpg"))