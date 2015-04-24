__author__ = 'tbeltramelli'

import cv2
from pylab import *


class UMedia:

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
    def load_videos(paths, callback):
        captures = []

        for i, path in enumerate(paths):
            cap = cv2.VideoCapture(path)
            captures.append(cap)

        is_reading = True
        while is_reading:
            imgs = []

            for i, cap in enumerate(captures):
                is_reading, img = cap.read()
                imgs.append(img)

            if is_reading == True:
                callback(imgs)

                ch = cv2.waitKey(33)
                if ch == 32:
                    cv2.destroyAllWindows()
                    break

    @staticmethod
    def get_frame_from_video(path, frame_number):
        cap = cv2.VideoCapture(path)
        frame_counter = 0
        is_reading = True

        while is_reading:
            is_reading, img = cap.read()
            frame_counter += 1

            if frame_counter == frame_number:
                return img

    @staticmethod
    def load_media(path, callback):
        if ".jpg" in path or ".png" in path or ".bmp" in path:
            callback(UMedia.get_image(path))
        else:
            UMedia.load_video(path, callback)

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
    def combine_images(image1, image2, scale=1.0):
        height, width = image1.shape[:2]
        width = int(width * scale)
        height = int(height * scale)

        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        image[:height, :width] = image1
        image[:height, width:width * 2] = image2

        return image