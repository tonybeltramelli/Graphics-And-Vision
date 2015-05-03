__author__ = 'tbeltramelli'

import cv2
import numpy as np

media_path = "../data/media/"
pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

object_points = []
left_points = []
right_points = []

left_camera_matrix = None
left_dist_coeff = None
right_camera_matrix = None
right_dist_coeff = None


def get_combined_image(left_img, right_img, scale=1.0, use_horizontally=True):
    height, width = left_img.shape[:2]
    width = int(width * scale)
    height = int(height * scale)

    left_img = cv2.resize(left_img, (width, height))
    right_img = cv2.resize(right_img, (width, height))

    if use_horizontally:
        image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        image[:height, :width] = left_img
        image[:height, width:width * 2] = right_img
    else:
        image = np.zeros((height * 2, width, 3), dtype=np.uint8)
        image[:height, :width] = left_img
        image[height:height * 2, :width] = right_img

    return image


def show(left_img, right_img):
    img = get_combined_image(left_img, right_img)
    img = cv2.pyrDown(img)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", img)
    cv2.waitKey(0)


def calibrate(left_img, right_img, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2):
    ret_left, corners_left = cv2.findChessboardCorners(left_img, pattern_size)
    ret_right, corners_right = cv2.findChessboardCorners(right_img, pattern_size)

    is_detected = ret_left and ret_right

    if not is_detected:
        return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2

    cv2.cornerSubPix(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), corners_left, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.01))
    cv2.cornerSubPix(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), corners_right, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.01))

    #######
    #cv2.drawChessboardCorners(left_img, pattern_size, corners_left, True)
    #cv2.drawChessboardCorners(right_img, pattern_size, corners_right, True)

    #show(left_img, right_img)
    #######

    height, width, layers = left_img.shape
    size = (width, height)

    object_points.append(pattern_points)
    left_points.append(corners_left.reshape(-1, 2))
    right_points.append(corners_right.reshape(-1, 2))

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH


    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points,
                                                                                                     left_points,
                                                                                                     right_points,
                                                                                                     cameraMatrix1,
                                                                                                     distCoeffs1,
                                                                                                     cameraMatrix2,
                                                                                                     distCoeffs2,
                                                                                                     size,
                                                                                                     criteria=criteria,
                                                                                                     flags=flags)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1,
                                                                      distCoeffs1,
                                                                      cameraMatrix2,
                                                                      distCoeffs2,
                                                                      size, R, T)

    map_left1, map_left2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, size, cv2.CV_16SC2)
    map_right1, map_right2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, size, cv2.CV_16SC2)

    new_left_img = cv2.remap(left_img, map_left1, map_left2, cv2.INTER_LINEAR)
    new_right_img = cv2.remap(right_img, map_right1, map_right2, cv2.INTER_LINEAR)

    show(new_left_img, new_right_img)

    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2

a = [25, 37, 74, 80, 113]
size = None

for index in a:
    left_img = cv2.imread(media_path + "l" + str(index) + ".jpg")
    right_img = cv2.imread(media_path + "r" + str(index) + ".jpg")

    if size is None:
        height, width, layers = left_img.shape
        size = (width, height)

    left_camera_matrix, left_dist_coeff, right_camera_matrix, right_dist_coeff = calibrate(left_img, right_img, left_camera_matrix, left_dist_coeff, right_camera_matrix, right_dist_coeff)

def process(images):
    left_img = images[0]
    right_img = images[1]

    left_camera_matrix = None
    left_dist_coeff = None
    right_camera_matrix = None
    right_dist_coeff = None

    left_camera_matrix, left_dist_coeff, right_camera_matrix, right_dist_coeff = calibrate(left_img, right_img, left_camera_matrix, left_dist_coeff, right_camera_matrix, right_dist_coeff)

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

#load_videos([media_path + "cameraLeft2.mov", media_path + "cameraRight2.mov"], process)