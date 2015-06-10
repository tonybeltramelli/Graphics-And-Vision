#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : StereoCameras.py                                         -->
#<!-- Description: Class used for managing the internal cameras parameters  -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 07/04/2015                                               -->
#<!-- Change     : 07/04/2015 - Creation of these classes                   -->
#<!-- Review     : 07/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040701 $'

########################################################################
class CamerasParameters(object):
    """CamerasParameters Class is used for managing the stereo cameras settings."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def CameraMatrix1(self):
        """Get the first camera matrix."""
        return self.__cameraMatrix1

    @CameraMatrix1.setter
    def CameraMatrix1(self, value):
        """Get the first camera matrix."""
        self.__cameraMatrix1 = value

    @property
    def CameraMatrix2(self):
        """Get the second camera matrix."""
        return self.__cameraMatrix2

    @CameraMatrix2.setter
    def CameraMatrix2(self, value):
        """Get the second camera matrix."""
        self.__cameraMatrix2 = value

    @property
    def DistCoeffs1(self):
        """Get the first camera distortion parameters."""
        return self.__distCoeffs1

    @DistCoeffs1.setter
    def DistCoeffs1(self, value):
        """Set the first camera distortion parameters."""
        self.__distCoeffs1 = value

    @property
    def DistCoeffs2(self):
        """Get the second camera distortion parameters."""
        return self.__distCoeffs2

    @DistCoeffs2.setter
    def DistCoeffs2(self, value):
        """Set the second camera distortion parameters."""
        self.__distCoeffs2 = value

    @property
    def R1(self):
        """Get a 3x3 rectification transform (rotation matrix) for the first camera."""
        return self.__r1

    @R1.setter
    def R1(self, value):
        """Set a 3x3 rectification transform (rotation matrix) for the first camera."""
        self.__r1 = value

    @property
    def R2(self):
        """Get a 3x3 rectification transform (rotation matrix) for the second camera."""
        return self.__r2

    @R2.setter
    def R2(self, value):
        """Set a 3x3 rectification transform (rotation matrix) for the second camera."""
        self.__r2 = value

    @property
    def P1(self):
        """Get a 3x4 projection matrix in the new (rectified) coordinate systems for the first camera."""
        return self.__p1

    @P1.setter
    def P1(self, value):
        """Set a 3x4 projection matrix in the new (rectified) coordinate systems for the first camera."""
        self.__p1 = value

    @property
    def P2(self):
        """Get a 3x4 projection matrix in the new (rectified) coordinate systems for the second camera."""
        return self.__p2

    @P2.setter
    def P2(self, value):
        """Set a 3x4 projection matrix in the new (rectified) coordinate systems for the second camera."""
        self.__p2 = value

    @property
    def Map1(self):
        """Get the left output map."""
        return self.__map1

    @Map1.setter
    def Map1(self, value):
        """Set the left output map."""
        self.__map1 = value

    @property
    def Map2(self):
        """Get the right output map."""
        return self.__map2

    @Map2.setter
    def Map2(self, value):
        """Set the right output map."""
        self.__map2 = value

    @property
    def Q(self):
        """Get a 4x4 disparity-to-depth mapping matrix."""
        return self.__q

    @Q.setter
    def Q(self, value):
        """Set a 4x4 disparity-to-depth mapping matrix."""
        self.__q = value

    @property
    def E(self):
        """Get the essential matrix."""
        return self.__e

    @E.setter
    def E(self, value):
        """Get the essential matrix."""
        self.__e = value

    @property
    def F(self):
        """Get the fundamental matrix."""
        return self.__f

    @F.setter
    def F(self, value):
        """Get the fundamental matrix."""
        self.__f = value

    #----------------------------------------------------------------------#
    #                  CamerasParameters Class Constructor                 #
    #----------------------------------------------------------------------#
    def __init__(self):
        """CamerasParameters Class Constructor."""
        self.Clear()

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def Clear(self):
        """Define the default values for all class attributes."""
        self.CameraMatrix1 = self.CameraMatrix2 = None
        self.DistCoeffs1   = self.DistCoeffs2   = None
        self.R1 = self.P1  = self.R2 = self.P2  = None
        self.Q  = self.E   = self.F  = None
