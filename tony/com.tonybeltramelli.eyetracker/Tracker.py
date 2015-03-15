__author__ = 'tbeltramelli'

from Eye import *
from UMedia import *


class Tracker:
    def __init__(self, path):
        e = Eye("../Sequences/right_corner.jpg", "../Sequences/left_corner.jpg")
        #Utils.load_video(path, e.process)
        e.process(UMedia.get_image("../Sequences/fakeeye.jpg"))
