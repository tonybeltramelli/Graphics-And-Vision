#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Camera.py                                                -->
#<!-- Description: Class used for managing the cameras connected in the     -->
#<!--              computer                                                 -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 05/04/2015                                               -->
#<!-- Change     : 05/04/2015 - Creation of these classes                   -->
#<!-- Review     : 05/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040501 $'

########################################################################
import cv2
from Settings import Constant as C

########################################################################
class ExceptionError(RuntimeError):
    """ExceptionError class raises when an error is detected that doesn't fall in any of the other exception categories.
       The associated value is a string indicating what precisely went wrong.
       SIGB: You don't need to modify this class."""
    pass

########################################################################
class Camera(object):
    """Camera class represents each individual camera (left or right) used in a stereo calibrated setup.
       SIGB: You are going to develop the main operations performed by a video capture device in OpenCV."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Index(self):
        """Get the index used for managering the video camera device."""
        return self.__index

    @property
    def Width(self):
        """Get the current width of captured images."""
        return int(self.__camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    @Width.setter
    def Width(self, value):
        """Set a new width value to captured images."""
        self.__camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(value))

    @property
    def Height(self):
        """Get the current height of captured images."""
        return int(self.__camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    @Height.setter
    def Height(self, value):
        """Set a new height value to captured images."""
        self.__camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(value))

    @property
    def Size(self):
        """Get the current size of captured images."""
        return (self.Width, self.Height)

    @Size.setter
    def Size(self, value):
        """Set a new size to captured images."""
        self.Width, self.Height = map(int, value)

    #----------------------------------------------------------------------#
    #                       Camera Class Constructor                       #
    #----------------------------------------------------------------------#
    def __init__(self, index):
        """Camera Class Constructor."""
        # Create an instance of a video capture device based on index argument.
        # SIGB: Use a private attribute called "self.__camera".


        # load video
        if index < 2:
            vid = C.VIDEO_LEFT_1
        else:
            vid = C.VIDEO_RIGHT_1

        # self.__camera = cv2.VideoCapture(index)
        # vid = cv2.VideoCapture(vid)
        # is_reading, img = vid.read()
        # vid = cv2.pyrDown(img)
        # self.__camera = vid
        self.__camera = cv2.VideoCapture(vid)

        # Save the index value in the "self.Index" property.
        # SIGB: This class does not have a function to write directly in "self.Index" property.
        #       For this reason, you should to write the value using the private attribute "self.__index".
        self.__index = index

        # Check if the video capture device has been initialized already using "isOpened()" function.
        # SIGB: Otherwise, throw an exception with a message about a problem in the camera initialization process.
        if not self.__camera.isOpened():
            raise ExceptionError("There was a failed during initialization process for camera index {}".format(self.Index))

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Grab(self):
        """Grabs the next frame from video file or capturing device."""
        # SIBG: Case there is any problem during this process, throw an exception with a message about the grab problem.
        if not self.__camera.grab():
            raise ExceptionError("There was a failed during grab process for camera index {}".format(self.Index))

    def Retrieve(self):
        """Decodes and returns the grabbed video frame."""
        # Get the grabbed image.
        retval, image = self.__camera.retrieve()

        # SIBG: Case there is any problem during this process, throw an exception with a message about the retrieve problem.
        if not retval:
            raise ExceptionError("There was a failed during retrieve process for camera index {}".format(self.Index))

        # Return the grabbed image.
        return image

    def Read(self):
        """Grabs, decodes and returns the next video frame."""
        # Get the grabbed image.
        retval, image = self.__camera.read()

        # SIBG: Case there is any problem during this process, throw an exception with a message about the retrieve problem.
        if not retval:
            raise ExceptionError("There was a failed during read image process for camera index {}".format(self.Index))

        # Return the grabbed image.
        return image

    def Release(self):
        """Closes video file or capturing device."""
        self.__camera.release()
