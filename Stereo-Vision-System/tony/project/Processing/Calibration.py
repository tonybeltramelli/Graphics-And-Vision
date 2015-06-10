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

__version__ = '$Revision: 2015040901 $'

########################################################################
import cv2
import numpy as np

from Cameras.CameraEnum    import CameraEnum
from Cameras.StereoCameras import StereoCameras

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

        # Defines the criterias used by stereoCalibrate() function.
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags  = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

        # <010> Calibrates a stereo camera setup.

        # <010> Records the external parameters.

        # <014> Return the final result.

    def CrossProductMatrix(self, T):
        """Estimating the skew symmetric matrix."""
        # <011> Estimate the skew symmetric matrix
        pass

    def EssentialMatrix(self, R, T):
        """Calculate the Essential Matrix."""
        # <012> Estimate manually the essential matrix.
        pass

    def FundamentalMatrix(self, K1, K2, E):
        """Calculate the Fundamental Matrix."""
        # <013> Estimate manually the fundamental matrix.
        pass

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