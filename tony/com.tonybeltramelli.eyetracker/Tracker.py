__author__ = 'tbeltramelli'

from Eye import *
from Utils import *


class Tracker:
    def __init__(self, path):
        e = Eye("../../bergar/Sequences/right_corner.jpg", "../../bergar/Sequences/left_corner.jpg")
        Utils.load_video(path, e.process)
        #e.process(Utils.get_image("../Sequences/Eye1.jpg"))