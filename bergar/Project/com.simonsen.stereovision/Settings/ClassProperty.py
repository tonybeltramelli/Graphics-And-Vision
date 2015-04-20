#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--                     EyeInfo - Eye Information Lab                     -->
#<!-- File       : ClassProperty.py                                         -->
#<!-- Description: Class for managing the direct access to non-instanced    -->
#<!--              objects in EyeInfo System                                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on an example available in Stack     -->
#<!--              Overflow Website (http://goo.gl/5YUJAQ)                  -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of this class                      -->
#<!-- Review     : 08/06/2014 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2014060801 $'

########################################################################
class ClassProperty(object):

    #----------------------------------------------------------------------#
    #                   ClassProperty Class Constructor                    #
    #----------------------------------------------------------------------#
    """Class for managing the direct access to non-instanced objects."""
    def __init__(self, getter, instance="0"):
        """ClassProperty Class Constructor."""
        self.getter   = getter
        self.instance = instance

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __get__(self, instance, owner):
        """Get the current object instance."""
        return self.getter(owner)
