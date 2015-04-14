from pylab import *
from numpy import *

class Camera:
    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center

    def project(self,X):
        """    Project points in X (4*n array) and normalize coordinates. """

        x = dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]

        #Translation (origin is considered to be at the center of the image but we want to transfer the origin to the corner of the image)
#        x[0]-=self.K[0][2]
#        x[1]-=self.K[1][2]
        return x

    def factor(self):
        """    Factorize the camera matrix into K,R,t as P = K[R|t]. """
        self.P=matrix(self.P)

        # factor first 3*3 part
        K,R = self.rq(self.P[:,:3])

        # make diagonal of K positive
        T = diag(sign(diag(K)))

        self.K = dot(K,T)
        self.R = dot(T,R) # T is its own inverse
        self.t = dot(inv(self.K),self.P[:,3])

        return self.K, self.R, self.t

    def center(self):
        """    Compute and return the camera center. """

        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T,self.t)
            return self.c

    def calibrate_from_points(self, x1,x2):
        return self.K

    def simple_calibrate(self, a,b):
        return self.K

    def rq(self, A):
        from scipy.linalg import qr

        Q,R = qr(flipud(A).T)
        R = flipud(R.T)
        Q = Q.T

        return R[:,::-1],Q[::-1,:]