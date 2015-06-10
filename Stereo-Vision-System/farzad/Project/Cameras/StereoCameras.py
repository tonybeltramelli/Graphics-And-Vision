#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : StereoCameras.py                                         -->
#<!-- Description: Class used for managing the stereo cameras               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: These classes are based on Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 05/04/2015                                               -->
#<!-- Change     : 05/04/2015 - Creation of these classes                   -->
#<!-- Review     : 05/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040501 $'

########################################################################
from Cameras import Cameras

from Setting.ClassProperty import ClassProperty

########################################################################
class StereoCameras(object):
    """CameraManager Class is used for managing some cameras instances."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for managing the stereo cameras."""
        if self.__Instance is None:
            self.__Instance = Cameras(0, 2)
        return self.__Instance
