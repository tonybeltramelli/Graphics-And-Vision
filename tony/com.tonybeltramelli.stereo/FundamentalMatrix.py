__author__ = 'tbeltramelli'

import cv2
from UMedia import *

class FundamentalMatrix:
    _points = []
    _img = None
    _raw_img = None

    _MAX_POINT_NUMBER = 16

    is_ready = False

    def get(self, img):
        self._img = img
        self._raw_img = copy(img)

        UMedia.show(self._img)
        cv2.setMouseCallback("image 0", self.mouse_event)

        print "Hit the space key when 8 points are selected in each image"
        cv2.waitKey(0)

        left = np.array(self._points[::2])
        right = np.array(self._points[1::2])

        fundamental_matrix = cv2.findFundamentalMat(left, right)

        print fundamental_matrix[0]

        fm = self.compute_fundamental(left, right)

        print ("---")

        print fm


        return np.array(fundamental_matrix[0])

    def compute_fundamental(self, x1,x2):
        n = x1.shape[1]

        A = zeros((n,9))

        for i in range(n):
            A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i], x1[1,i]*x2[0,i],
            x1[1,i]*x2[1,i], x1[1,i]*x2[2,i], x1[2,i]*x2[0,i], x1[2,i]*x2[1,i],
            x1[2,i]*x2[2,i] ]
        U,S,V = linalg.svd(A)
        F = V[-1].reshape(3,3)
        U,S,V = linalg.svd(F)
        S[2] = 0
        F = dot(U,dot(diag(S),V))
        return F

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