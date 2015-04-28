#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Pattern.py                                               -->
#<!-- Description: Class used for recognizing the calibration pattern       -->
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

from CalibrationEnum import CalibrationEnum

########################################################################

class Pattern(object):
    """Pattern Class is used for recognizing the calibration pattern."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Type(self):
        """Get the pattern type used for calibrating the cameras."""
        return self.__type

    @Type.setter
    def Type(self, value):
        """Set the pattern type used for calibrating the cameras."""
        self.__type = value
        self.__size = (4, 11) if self.__type == CalibrationEnum.CIRCLES else (9, 6)

    @property
    def Size(self):
        """Get the pattern size used for calibrating the cameras."""
        return self.__size

    @property
    def LeftCorners(self):
        """Get the output array of detected left corners."""
        return self.__leftCorners

    @LeftCorners.setter
    def LeftCorners(self, value):
        """Set the output array of detected left corners."""
        self.__leftCorners = value

    @property
    def RightCorners(self):
        """Get the output array of detected right corners."""
        return self.__rightCorners

    @RightCorners.setter
    def RightCorners(self, value):
        """Set the output array of detected right corners."""
        self.__rightCorners = value

    @property
    def ObjectPoints(self):
        """Get the vector with the number of the pattern views."""
        return self.__objectPoints

    @ObjectPoints.setter
    def ObjectPoints(self, value):
        """Set the vector with the number of the pattern views."""
        self.__objectPoints = value

    #----------------------------------------------------------------------#
    #                       Pattern Class Constructor                      #
    #----------------------------------------------------------------------#
    def __init__(self, pattern=CalibrationEnum.CHESSBOARD):
        """Pattern Class Constructor."""
        self.Type = pattern
        self.Clear()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def FindCorners(self, image, isDrawed=True):
        """Find the pattern in the image."""
        # Checks what kind of pattern will be used for calibrating the cameras.
        if self.Type == CalibrationEnum.CHESSBOARD:
            corners = self.FindChessboardCorners(image)
        elif self.Type == CalibrationEnum.CIRCLES:
            corners = self.FindCirclesGrid(image)
        else:
            corners = None

        # Checks if the user wants to print the detected corners in the input image.
        if isDrawed and corners is not None:
            cv2.drawChessboardCorners(image, self.Size, corners, True)

        # return the detected corners.
        return corners

    def FindChessboardCorners(self, image):
        """Finds the positions of internal corners of the chessboard."""
        # Processed image
        gray = image.copy()

        # Finds the positions of internal corners of the chessboard.
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        retval, corners = cv2.findChessboardCorners(gray, self.Size, flags=flags)

        if retval:
            # Check if the input image is a grayscale image.
            if len(gray.shape) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            # Refines the corner locations.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            corners = None

        # Garbage Collector.
        del gray

        # Return the final result.
        return corners

    def FindCirclesGrid(self, image):
        """Finds centers in the grid of circles using an asymmetric pattern."""
        # Processed image
        gray = image.copy()

        # Finds centers in the grid of circles.
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        retval, corners = cv2.findCirclesGrid(gray, self.Size, flags=flags)

        # Garbage Collector.
        del gray

        # Return the final result.
        return corners

    def CalculatePattern(self):
        """Creates a standard vectors of the calibration pattern points."""
        # Check what kind of pattern the user will use during the camera calibration process.
        self.__size = (11, 4) if self.Type == CalibrationEnum.CIRCLES else (9, 6)

        # Create the main vector.
        objectPoints = np.zeros((np.prod(self.Size), 3), np.float32)
        objectPoints[:, :2] = np.indices(self.Size).T.reshape(-1, 2)

        # If the user will use a circle pattern, it is necessary to modify the main vector.
        if self.Type == CalibrationEnum.CIRCLES:
            objectPoints[:, 0] = objectPoints[:, 0] * 2.0 + objectPoints[:, 1] % 2
            self.__size = (4, 11)

        # Return the final result.
        return objectPoints

    def Clear(self):
        """Empty all internal parameters used for calibrating a stereo cameras setup."""
        self.LeftCorners  = []
        self.RightCorners = []
        self.ObjectPoints = []
