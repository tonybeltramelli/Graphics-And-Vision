#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Cameras.py                                               -->
#<!-- Description: Class used for managing the stereo cameras               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: These classes are based on Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 09/04/2015                                               -->
#<!-- Change     : 09/04/2015 - Creation of these classes                   -->
#<!-- Review     : 09/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040901 $'

########################################################################
from Camera            import Camera
from CameraEnum        import CameraEnum
from CamerasParameters import CamerasParameters

########################################################################
class Cameras(object):
    """Cameras Class is used for managing the stereo cameras."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Width(self):
        """Get the current width of captured images."""
        return self.__camera[CameraEnum.LEFT].Width

    @Width.setter
    def Width(self, value):
        """Set a new width value to captured images."""
        for index in self.__camera:
            self.__camera[index].Width = int(value)

    @property
    def Height(self):
        """Get the current height of captured images."""
        return self.__camera[CameraEnum.LEFT].Height

    @Height.setter
    def Height(self, value):
        """Set a new height value to captured images."""
        for index in self.__camera:
            self.__camera[index].Height = int(value)

    @property
    def Size(self):
        """Get the current size of captured images."""
        return self.__camera[CameraEnum.LEFT].Size

    @Size.setter
    def Size(self, value):
        """Set a new size to captured images."""
        for index in self.__camera:
            self.__camera[index].Size = value

    @property
    def Parameters(self):
        """Get the cameras parameters attributes."""
        return self.__parameters

    @Parameters.setter
    def Parameters(self, value):
        """Set the cameras parameters attributes."""
        self.__parameters = value

    #----------------------------------------------------------------------#
    #                       Cameras Class Constructor                      #
    #----------------------------------------------------------------------#
    def __init__(self, left=0, right=1, width=640, height=480):
        """Cameras Class Constructor."""
        # Creates a dictionary for managering the left and right cameras.
        self.__camera = {}
        # Creates a Camera object for each video capture device used in a stereo setup.
        self.__camera[CameraEnum.LEFT]  = Camera(left)
        self.__camera[CameraEnum.RIGHT] = Camera(right)

        # Define the standard size to captured images.
        self.Size = (width, height)

        # Define an object to manager the cameras parameters attributes.
        self.Parameters = CamerasParameters()

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def Grab(self):
        """Grabs the next frame from video file or capturing device."""
        for index in self.__camera:
            self.__camera[index].Grab()

    def Retrieve(self):
        """Decodes and returns the grabbed video frame."""
        return [self.__camera[index].Retrieve() for index in self.__camera]

    def Read(self):
        """Grabs, decodes and returns the next video frame."""
        return [self.__camera[index].Read() for index in self.__camera]

    def Release(self):
        """Closes video file or capturing device."""
        for index in self.__camera:
            self.__camera[index].Release()
