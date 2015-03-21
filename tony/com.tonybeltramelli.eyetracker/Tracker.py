__author__ = 'tbeltramelli'

from Eye import *
from UMedia import *


class Tracker:
    def __init__(self, path, right_template, left_template):
        e = Eye(right_template, left_template)

        if ".jpg" in path or ".png" in path:
            e.process(UMedia.get_image(path))
        else:
            UMedia.load_video(path, e.process)