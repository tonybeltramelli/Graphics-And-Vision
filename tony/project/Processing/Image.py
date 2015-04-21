#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Image.py                                                 -->
#<!-- Description: Class used for manipulating the 3D images                -->
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

from Cameras.StereoCameras import StereoCameras

########################################################################
class Image(object):
    """Image Class is used for manipulating the 3D images."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Disparity(self):
        """Get the disparity map."""
        return self.__disparity

    @Disparity.setter
    def Disparity(self, value):
        """Set the disparity map."""
        self.__disparity = value
    
    @property
    def Stereo(self):
        """Get the final stereo image."""
        return self.__stereo

    @Stereo.setter
    def Stereo(self, value):
        """Set the final stereo image."""
        self.__stereo = value

    @property
    def Left(self):
        """Get the image provided by the left camera."""
        return self.__left

    @Left.setter
    def Left(self, value):
        """Set the image provided by the left camera."""
        self.__left = value

    @property
    def Right(self):
        """Get the image provided by the right camera."""
        return self.__right

    @Right.setter
    def Right(self, value):
        """Set the image provided by the right camera."""
        self.__right = value

    #----------------------------------------------------------------------#
    #                        Image Class Constructor                       #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Image Class Constructor."""
        self.Clear()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def StereoImages(self, left, right):
        """Create the stereo images."""
        # Get transformation maps.
        map1 = StereoCameras.Instance.Parameters.Map1
        map2 = StereoCameras.Instance.Parameters.Map2

        # Applies a generic geometrical transformation to each stereo image.
        self.Left  = cv2.remap(left,  map1[0], map1[1], cv2.INTER_LINEAR)
        self.Right = cv2.remap(right, map2[0], map2[1], cv2.INTER_LINEAR)

    def StereoSGBM(self, minDisparity=0, blockSize=1):
        """Computing a stereo correspondence using the block matching algorithm."""
        # SIGB: All values used in this function were informed by OpenCV docs.
        sgbm = cv2.StereoSGBM_create(minDisparity, minDisparity + 16, blockSize,
                                     P1=8*3*blockSize**2, P2=32*3*blockSize**2, 
                                     disp12MaxDiff=1, preFilterCap=63,
                                     uniquenessRatio=10, speckleWindowSize=100,
                                     speckleRange=32, mode=cv2.STEREO_SGBM_MODE_HH)

        # Computes disparity map for the specified stereo pair
        self.Disparity = sgbm.compute(self.Left, self.Right).astype(np.float32) / 16.0

    def Clear(self):
        """Empty all internal parameters used for calibrating a stereo cameras setup."""
        self.Disparity = np.zeros((1, 1, 3), dtype=np.uint8)
        self.Stereo    = np.zeros((1, 1, 3), dtype=np.uint8)
        self.Left      = np.zeros((1, 1, 3), dtype=np.uint8)
        self.Right     = np.zeros((1, 1, 3), dtype=np.uint8)
