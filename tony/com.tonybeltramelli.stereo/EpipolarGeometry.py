__author__ = 'tbeltramelli'

import cv2
from UMedia import *

class EpipolarGeometry:
    _points = []
    _img = None
    _raw_img = None
    _fundamental_matrix = None
    _epipole = None

    _MAX_POINT_NUMBER = 16

    is_ready = False

    def __init__(self, img):
        self._img = img
        self._raw_img = copy(img)

        UMedia.show(self._img)
        cv2.setMouseCallback("image 0", self.mouse_event)

        print "Hit the space key when 8 points are selected in each image"
        cv2.waitKey(0)

        left = np.array(self._points[::2])
        right = np.array(self._points[1::2])

        #self._fundamental_matrix = self.compute_fundamental_matrix(left, right)
        #self._epipole = self.compute_epipole(self._fundamental_matrix)

        #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

        #point = [100, 100, 1]
        #self.show_epipolar_lines(self._raw_img, point)

        F, mask = cv2.findFundamentalMat(left, right)

        lines1 = cv2.computeCorrespondEpilines(right.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        self.drawlines(self._raw_img,lines1,left,right)

        height, width, layers = self._raw_img.shape

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(left.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        self.drawlines(self._raw_img, lines2,right,left, width/2)

        UMedia.show(self._raw_img)
        cv2.waitKey(0)

    def drawlines(self, img,lines,pts1,pts2, shift=0):
        r,c,l = img.shape
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            cv2.line(img, (x0 + shift,y0), (x1 + shift,y1), color,1)
            cv2.circle(img, (pt1[0] + shift, pt1[1]),5,color)
            cv2.circle(img, (pt2[0] + shift, pt2[1]),5,color)

        return img


    def compute_fundamental_matrix(self, x1, x2):
        n = x1.shape[1]
        a = zeros((n, 9))

        for i in range(n):
            a[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                    x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                    x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

        u, s, v = linalg.svd(a)
        f = v[-1].reshape(3, 3)
        u, s, v = linalg.svd(f)
        s[2] = 0
        f = dot(u, dot(diag(s), v))
        return f

    def compute_epipole(self, f):
        u, s, v = linalg.svd(f)
        e = v[-1]
        return e/e[2]

    def show_epipolar_lines(self, img, point):
        m, n = img.shape[:2]
        line = dot(self._fundamental_matrix, point)

        t = linspace(0, n, 100)
        lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

        print lt

    def mouse_event(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_ready:
                return

            height, width, layers = self._img.shape

            point = (x, y)

            color = (0, 0, 255) if len(self._points) % 2 == 0 else (255, 0, 0)

            cv2.circle(self._img, point, 3, color, thickness=-1)

            point = (point[0] - (0 if len(self._points) % 2 == 0 else width / 2), point[1])

            point = (point[0], point[1], 1)
            self._points.append(point)

            if len(self._points) is self._MAX_POINT_NUMBER:
                print "done"
                self.is_ready = True

            UMedia.show(self._img)
        elif event == cv2.EVENT_RBUTTONUP:
            self._points = []
            self._img = copy(self._raw_img)

            UMedia.show(self._img)