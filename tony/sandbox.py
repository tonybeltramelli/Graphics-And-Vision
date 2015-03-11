import cv2
import numpy as np

from assignment1.tony import SIGBWindows


def im1(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    cimg = img.copy()

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)

    if circles is not None:
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("result", cimg)
    cv2.waitKey(0)

def im2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 255, 0), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def im3(img):
    img = cv2.medianBlur(img, 5)
    cimg =  cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 9, 20, param1=50, param2=200, minRadius=40, maxRadius=110)

    if circles is not None:
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("result", cimg)
    cv2.waitKey(0)

def hough(windows):
    def houghCallback(image, sliderValues):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gray = get_binary(gray, 70)

        dp = int(sliderValues['hough_dp'])
        minDist = int(sliderValues['hough_min_dist'])
        param1 = int(sliderValues['hough_param1'])
        param2 = int(sliderValues['hough_param2'])
        minRadius = int(sliderValues['hough_min_radius'])
        maxRadius = int(sliderValues['hough_max_radius'])

    #    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, None, 10, 350, 50, 155)

#        gray = cv2.Canny(gray, 100, 128)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, None, param1, param2, minRadius, maxRadius)

        result = image
        if circles is not None:
            circles = circles[0]
            for circle in circles:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                color = (0, 255, 0)
                cv2.circle(result, center, 3, (0, 255, 100))
                cv2.circle(result, center, radius, color)


            circle = circles[0, :]
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            color = (0, 0, 255)
            cv2.circle(result, center, 3, color)
            cv2.circle(result, center, radius, color, 5)

        return result

    windows.registerSlider("hough_dp", 8, 15)
    windows.registerSlider("hough_min_dist", 307, 500)
    windows.registerSlider("hough_param1", 52, 200)
    windows.registerSlider("hough_param2", 447, 1500)
    windows.registerSlider("hough_min_radius", 28, 500)
    windows.registerSlider("hough_max_radius", 110, 500)
    windows.registerOnUpdateCallback("hough", houghCallback, "Temp")

def get_binary(img, threshold):
    matrix = img

    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            matrix[i][j] = 0 if matrix[i][j] > threshold else 255

    return matrix

#src = cv2.imread("Sequences/eye1.jpg")
#src = get_binary(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), 70)

#im1(src)
#im2(src)
#im3(src)

windows = SIGBWindows(mode="video")
#windows.openImage("Sequences/eye1.jpg")
windows.openVideo("Sequences/eye15.mp4")
hough(windows)
windows.show()


from scipy.cluster.vq import *

class Detection:
    def detectPupilKMeans(self, gray, K=2, distanceWeight=2, reSize=(40, 40)):
        smallI = cv2.resize(gray, reSize)
        M, N = smallI.shape

        X, Y = np.meshgrid(range(M), range(N))

        z = smallI.flatten()
        x = X.flatten()
        y = Y.flatten()
        O = len(x)

        features = np.zeros((O, 3))
        features[:, 0] = z
        features[:, 1] = y / distanceWeight
        features[:, 2] = x / distanceWeight
        features = np.array(features, 'f')

        centroids, variance = kmeans(features, K)

        label, distance = vq(features, centroids)

        return np.array(np.reshape(label, (M, N)))

    def detectPupilHough(self, gray):
        blur = cv2.GaussianBlur(gray, (9, 9), 3)

        dp = 6
        minDist = 10
        highThr = 30
        accThr = 600
        maxRadius = 70
        minRadius = 20

        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, minDist, None, highThr, accThr, minRadius, maxRadius)

        gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if (circles != None):
            print("detected")
            all_circles = circles[0]
            M, N = all_circles.shape
            k = 1
            for c in all_circles:
                cv2.circle(gColor, (int(c[0]), int(c[1])), c[2], (int(k * 255 / M), k * 128, 0))
                K = k + 1

            c = all_circles[0, :]
            cv2.circle(gColor, (int(c[0]), int(c[1])), c[2], (0, 0, 255))
        else:
            print("not detected")
        return gColor
