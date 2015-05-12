#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Calibration.py                                           -->
#<!-- Description: Class used for performing the calibration process        -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 09/04/2015                                               -->
#<!-- Change     : 09/04/2015 - Creation of these classes                   -->
#<!-- Review     : 09/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->
from sympy.printing import dot

__version__ = '$Revision: 2015040901 $'

########################################################################
import cv2
import numpy as np

from pylab import linalg

import Configuration

from Cameras.CameraEnum         import CameraEnum
from Cameras.StereoCameras      import StereoCameras
from Cameras.CamerasParameters  import CamerasParameters
# from Processing.Configuration   import Configuration

########################################################################
class Calibration(object):
    """Calibration Class is used for calibrating the stereo cameras."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def IsCalibrated(self):
        """Check if the stereo cameras are calibrated."""
        return self.__isCalibrated

    @IsCalibrated.setter
    def IsCalibrated(self, value):
        """Define that the stereo camera are calibrated or not."""
        self.__isCalibrated = value

    #----------------------------------------------------------------------#
    #                     Calibration Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Calibration Class Constructor."""
        self.IsCalibrated = False

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def CalibrateCamera(self, imagePoints, objectPoints, camera=CameraEnum.LEFT):
        """Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern."""
        # Prepares the parameters.
        imageSize = StereoCameras.Instance.Size
    
        # Output 3x3 floating-point camera matrix and output vector of distortion coefficients.
        cameraMatrix = np.zeros((3, 3))
        distCoeffs   = np.zeros((5, 1))
    
        # Calibrates a single camera.
        _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs)
    
        # Return the final result
        return cameraMatrix, distCoeffs
    
    def StereoCalibrate(self, leftCorners, rightCorners, objectPoints):
        """Calibrates the stereo camera."""
        # <009> Prepares the external parameters.
        imagePointsLeft         = Configuration.Configuration.Instance.Pattern.LeftCorners
        imagePointsRight        = Configuration.Configuration.Instance.Pattern.RightCorners
        imageSize               = StereoCameras.Instance.Size

        cameraMatrixLeft        = StereoCameras.Instance.Parameters.CameraMatrix1
        cameraMatrixRight       = StereoCameras.Instance.Parameters.CameraMatrix2
        distCoeffLeft           = StereoCameras.Instance.Parameters.DistCoeffs1
        distCoeffRight          = StereoCameras.Instance.Parameters.DistCoeffs2


        # Defines the criterias used by stereoCalibrate() function.
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags  = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

        # <010> Calibrates a stereo camera setup.
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight, imageSize, cameraMatrixLeft, cameraMatrixRight, distCoeffLeft, distCoeffRight)

        E2 = self.EssentialMatrix(R, T)
        F2 = self.FundamentalMatrix(cameraMatrix1, cameraMatrix2, E2)


        # TODO: Store other values?

        # <010> Records the external parameters.
        StereoCameras.Instance.Parameters.CameraMatrix1 = cameraMatrix1
        StereoCameras.Instance.Parameters.CameraMatrix2 = cameraMatrix2
        StereoCameras.Instance.Parameters.DistCoeffs1 = distCoeffs1
        StereoCameras.Instance.Parameters.DistCoeffs2 = distCoeffs2

        # <014> Return the final result.
        return R, T

    def CrossProductMatrix(self, T):
        """Estimating the skew symmetric matrix."""
        # <011> Estimate the skew symmetric matrix
        sm = [
            [0, -T[2], T[1]],
            [T[2], 0, -T[0]],
            [-T[1], T[0], 0]
        ]
        return sm

    def EssentialMatrix(self, R, T):
        """Calculate the Essential Matrix."""
        # <012> Estimate manually the essential matrix.
        sm = self.CrossProductMatrix(T)
        E = sm * R
        return E

    def FundamentalMatrix(self, K1, K2, E):
        """Calculate the Fundamental Matrix."""
        # <013> Estimate manually the fundamental matrix.
        K2_new = linalg.transpose(linalg.inv(K2))
        K1_new = linalg.inv(K1)
        F = K2_new * E * K1_new
        return F

    def StereoRectify(self, R, T):
        """Computes rectification transforms for each head of a calibrated stereo camera."""
        # Prepares the external parameters.
        cameraMatrix1 = StereoCameras.Instance.Parameters.CameraMatrix1
        distCoeffs1   = StereoCameras.Instance.Parameters.DistCoeffs1
        cameraMatrix2 = StereoCameras.Instance.Parameters.CameraMatrix2
        distCoeffs2   = StereoCameras.Instance.Parameters.DistCoeffs2

        imageSize = StereoCameras.Instance.Size

        # Computes rectification transforms.
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, alpha=0)

        # Records the external parameters.
        StereoCameras.Instance.Parameters.R1 = R1
        StereoCameras.Instance.Parameters.R2 = R2
        StereoCameras.Instance.Parameters.P1 = P1
        StereoCameras.Instance.Parameters.P2 = P2
        StereoCameras.Instance.Parameters.Q  = Q

        # Define that the stereo cameras are calibrated
        self.IsCalibrated = True

    def UndistortRectifyMap(self):
        """Computes the undistortion and rectification transformation maps."""
        # Prepares the external parameters.
        cameraMatrix1 = StereoCameras.Instance.Parameters.CameraMatrix1
        distCoeffs1   = StereoCameras.Instance.Parameters.DistCoeffs1
        cameraMatrix2 = StereoCameras.Instance.Parameters.CameraMatrix2
        distCoeffs2   = StereoCameras.Instance.Parameters.DistCoeffs2

        R1 = StereoCameras.Instance.Parameters.R1
        R2 = StereoCameras.Instance.Parameters.R2
        P1 = StereoCameras.Instance.Parameters.P1
        P2 = StereoCameras.Instance.Parameters.P2

        imageSize = StereoCameras.Instance.Size

        # Computes the undistortion and rectification transformation maps
        map1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
        map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)

        # Records the external parameters.
        StereoCameras.Instance.Parameters.Map1 = map1
        StereoCameras.Instance.Parameters.Map2 = map2