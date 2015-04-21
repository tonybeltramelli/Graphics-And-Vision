#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Configuration.py                                         -->
#<!-- Description: Class used for configuring the stereo cameras            -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: These classes are based on Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 05/04/2015                                               -->
#<!-- Change     : 05/04/2015 - Creation of these classes                   -->
#<!--            : 07/04/2015 - Comentaries                                 -->
#<!-- Review     : 07/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040701 $'

########################################################################
from assignment3.tony.project.Processing import Config

from Setting.ClassProperty import ClassProperty

########################################################################    
class Configuration(object):
    """Configuration Class is used for configuring the stereo cameras."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for configuring the stereo cameras."""
        if self.__Instance is None:
            self.__Instance = Config()
        return self.__Instance
