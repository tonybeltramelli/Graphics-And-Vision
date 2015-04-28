#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Config.py                                                -->
#<!-- Description: Class used for configuring the stereo cameras            -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 09/04/2015                                               -->
#<!-- Change     : 09/04/2015 - Creation of these classes                   -->
#<!--            : 18/04/2015 - Add the augmented object                    -->
#<!-- Review     : 09/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015041801 $'

########################################################################
from Augmented       import Augmented
from Calibration     import Calibration
from CalibrationEnum import CalibrationEnum
from Image           import Image
from Pattern         import Pattern

########################################################################
class Config(object):
    """Config Class is used for configuring the stereo cameras."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Augmented(self):
        """Get the augmente object for performin the augmented reality process."""
        return self.__augmented

    @property
    def Calibration(self):
        """Get the calibration object for performing the stereo cameras calibration process."""
        return self.__calibration

    @property
    def Image(self):
        """Get the object used for managering the stereo images."""
        return self.__image

    @property
    def Pattern(self):
        """Get the pattern object used during the calibration process."""
        return self.__pattern

    #----------------------------------------------------------------------#
    #                    Configuration Class Constructor                   #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Configuration Class Constructor."""
        self.__augmented   = Augmented()
        self.__calibration = Calibration()
        self.__image       = Image()
        self.__pattern     = Pattern(CalibrationEnum.CHESSBOARD)

    def Clear(self):
        """Empty all internal parameters used for calibrating a stereo cameras setup."""
        self.Calibration.IsCalibrated = False
        self.Image.Clear()
        self.Pattern.Clear()
