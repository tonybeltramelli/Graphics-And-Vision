__author__ = 'tbeltramelli'

import cv2
from pylab import *

class Utils:

    @staticmethod
    def get_image(path):
        return cv2.imread(path)

    @staticmethod
    def load_video(path, callback):
        cap = cv2.VideoCapture(path)
        is_reading = True
        while is_reading:
            is_reading, img = cap.read()
            if is_reading == True:
                callback(img)

                ch = cv2.waitKey(33)
                if ch == 32:
                    cv2.destroyAllWindows()
                    break

    @staticmethod
    def show(*images):
        for i, img in enumerate(images):
            cv2.namedWindow(("image %d" % i), cv2.WINDOW_AUTOSIZE)
            cv2.imshow(("image %d" % i), img)
            #cv2.waitKey(0)
            #cv2.destroyWindow(("image %d" % i))

    @staticmethod
    def show_all_gray(*images):
        for i, img in enumerate(images):
            gray()
            subplot(1, len(images), i+1)
            title(("image %d" % i))
            imshow(img)
        show()

    @staticmethod
    def show_all_rgb(*images):
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            subplot(1, len(images), i+1)
            title(("image %d" % i))
            imshow(img)
        show()

    @staticmethod
    def normalize(range_min, range_max, x, x_min, x_max):
        return range_min + (((x - x_min) * (range_max - range_min)) / (x_max - x_min))

    @staticmethod
    def is_in_area_center(x, y, width, height):
        return ((x > width/4) and (x < 3 * (width/4))) and ((y > height/4) and (y < 3 * (height/4)))