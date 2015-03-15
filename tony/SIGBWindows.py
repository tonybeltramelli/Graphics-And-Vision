import cv2
import numpy as np

class SIGBWindows:
    '''
    Window Manipulation Class

    Acts as a wrapper around cv2 windowing functions
    Creates 3 windows: Settings, Results and Temp
    Settings window is used to show sliders (trackbars)
    Results window is used to show total results (not really used)
    Temp window is used to show temporary results
    '''
    def __init__(self, mode="video"):
        '''
        Creates the windows, registers mode

        Parameters:
            mode: ("video", "image", "cam") selects input to be used

        Returns:
            class instance
        '''
        self.updateCallbacks = dict()
        self.sliders = []
        self.mode = mode
        cv2.namedWindow("Settings")
        cv2.namedWindow("Results")
        cv2.namedWindow("Temp")

    def show(self):
        '''
        Shows all the windows with the right size and position
        Initiates the cam, etc.

        '''
        cv2.resizeWindow("Settings", 1000, 450)
        cv2.moveWindow("Settings", 300, 540)

        cv2.resizeWindow("Results", 640, 480)
        cv2.moveWindow("Results", 0, 0)

        cv2.resizeWindow("Temp", 640, 480)
        cv2.moveWindow("Temp", 1030, 0)

        if self.mode == "cam":
            while True:
                key = cv2.waitKey(1)

                self.image = self.getVideoStreamCam()
                self.update()

                if key == 0:
                    break
        else:
            cv2.setTrackbarPos("video_position", "Settings", 1)
            sliderValues = self.getSliderValues()
            self.image = self.getVideoFrame(sliderValues['video_position'])
            self.update()
            key = cv2.waitKey(0)

        cv2.destroyAllWindows()


    def showCam(self):
        '''
        Shows the cam windows
        '''
        cv2.resizeWindow("Settings", 1000, 450)
        cv2.moveWindow("Settings", 300, 540)

        cv2.resizeWindow("Results", 640, 480)
        cv2.moveWindow("Results", 0, 0)

        cv2.resizeWindow("Temp", 640, 480)
        cv2.moveWindow("Temp", 1030, 0)


        self.update()

        key = cv2.waitKey(0)

    def update(self, trackbarPos=None):
        '''
        This gets called when trackbars get changed or the windows need to be redrawn

        '''
        sliderValues = self.getSliderValues()

        if self.mode == "video":
            sliderValues = self.getSliderValues()
            self.image = self.getVideoFrame(sliderValues['video_position'])

        image = np.copy(self.image)

        cv2.imshow("Results", image)
        cv2.imshow("Temp", image)

        for callbackName in self.updateCallbacks:
            callback = self.updateCallbacks[callbackName]
            func = callback['function']
            window = callback['window']
            result = func(image, sliderValues)

            x = 10
            y = 20
            for slider in sliderValues:
                value = slider + ": " + str(sliderValues[slider])
#                cv2.putText(result, value, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
                cv2.putText(result, value, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                y = y + 20

            cv2.imshow(window, result)

    def registerSlider(self, name, startingValue, maxValue):
        '''
        Register a new slider

        min value is implicitly 0

        Params:
            name (string): name used to refer to the slider, will appear in the
                           sliderValues dict as a key
            startinValue (int): the default value the slider should start with
            maxValue (int): the max value of the slider
        '''
        self.sliders.append(name)
        cv2.createTrackbar(name, "Settings", startingValue, maxValue, self.update)

    def deregisterSlider(self, name):
        '''
        Removes slider from window

        Params:
            name (string): name of the slider to remove
        '''
        self.sliders = [slider for slider in self.sliders if slider != name]

    def getSliderValues(self):
        '''
        Retrieve values for all registered sliders

        Returns
            dict slider_name -> value
        '''
        values = dict()
        for slider in self.sliders:
            values[slider] = cv2.getTrackbarPos(slider, "Settings")
        return values

    def registerOnUpdateCallback(self, name, function, window="Results"):
        '''
        Registers a callback function to be called when sliders get updated

        Parameters:
            name (string): name of the callback function
            function (function): the callback (should accept one param, a dict of slider name -> value pairs)
            window (string): which window name it should show up in
        '''
        self.updateCallbacks[name] = {
                                      'function': function,
                                      'window': window
                                      }

    def openVideo(self, videoFile):
        '''
        Opens video for reading

        Params:
            videoFile (string): path to video file to be open
        '''
        self.videoFile = videoFile
        self.video = cv2.VideoCapture(videoFile)
        self.registerSlider("video_position", 2, self.getTotalVideoFrames())

    def openImage(self, imageFile):
        '''
        Open static image

        Params:
            imageFile (string): path to image
        '''
        self.image = cv2.imread(imageFile)


    def getVideoFrame(self, frameIndex):
        '''
        Load a frame from the currently open video

        Params:
            frameIndex (int): frame number in the video

        Returns:
            image read from the video (numpy array)
        '''
        frameIndex = min(frameIndex, self.getTotalVideoFrames() - 1)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        retval, image = self.video.read()
        return image

    def getVideoStreamCam(self):
        '''
        Read image from cam

        Returns:
            (numpy array) image from cam
        '''
        self.video = cv2.VideoCapture(1)
        retval, image = self.video.read()
        return image

    def getTotalVideoFrames(self):
        '''
        How many frames does the current video contain?

        Returns:
            (int) frame count
        '''
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def getVideoWriter(self, filename):
        size = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.videoWriter = cv2.VideoWriter(filename, cv2.cv.FOURCC("X", "V", "I", "D"), fps, size)

        return self.videoWriter

