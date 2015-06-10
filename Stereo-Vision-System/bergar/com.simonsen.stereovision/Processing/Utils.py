__author__ = 'bs'

import cv2

def get_frame_from_video(path, frame_number):
    cap = cv2.VideoCapture(path)
    frame_counter = 0
    is_reading = True

    while is_reading:
        is_reading, img = cap.read()
        frame_counter += 1

        if frame_counter == frame_number:
            return img