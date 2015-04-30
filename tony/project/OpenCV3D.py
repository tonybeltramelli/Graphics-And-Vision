#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : OpenCV3D.py                                              -->
#<!-- Description: Main class of this project                               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 05/04/2015                                               -->
#<!-- Change     : 05/04/2015 - Creation of these classes                   -->
#<!--            : 06/04/2015 - Comentaries                                 -->
#<!-- Review     : 06/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015040601 $'

########################################################################

from collections import deque

from Cameras.StereoCameras    import StereoCameras
from Processing.Configuration import Configuration

########################################################################
class OpenCV3D(object):
    """OpenCV3D class is the main class of this project."""
    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    ___plyHeader = '''ply
format ascii 1.0
element vertex %(num)d
property float x
property float y
property float z
end_header
'''

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Image(self):
        """Get the last processed image."""
        return self.__image

    @Image.setter
    def Image(self, value):
        """Set the last processed image."""
        self.__image = value

    @property
    def PointsQueue(self):
        """Get the queue with selected points."""
        return self.__pointsQueue

    @PointsQueue.setter
    def PointsQueue(self, value):
        """Set the queue with selected points."""
        self.__pointsQueue = value

    @property
    def IsCalibrating(self):
        """Check if the calibration process is running."""
        return self.__isCalibrating

    @IsCalibrating.setter
    def IsCalibrating(self, value):
        """Set that the calibration process starts."""
        self.__isCalibrating = value

    @property
    def IsSaving(self):
        """Check if the PLY save file process is running."""
        return self.__isSaving

    @IsSaving.setter
    def IsSaving(self, value):
        """Set that the PLY save process starts."""
        self.__isSaving = value

    @property
    def IsFrozen(self):
        """Check if the fundamental matrix process is running."""
        return self.__isFrozen

    @IsFrozen.setter
    def IsFrozen(self, value):
        """Set that the fundamental matrix process is running."""
        self.__isFrozen = value

    #----------------------------------------------------------------------#
    #                      OpenCV3D Class Constructor                      #
    #----------------------------------------------------------------------#
    def __init__(self):
        """OpenCV3D Class Constructor."""
        self.Clear()

    def __del__(self):
        """OpenCV3D Class Destructor."""    
        # Stops the main thread system.
        self.Stop()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Start(self):
        """Start a new thread for managering the system."""
        #self.__isRunning = True
        #self.__thread = Thread(target=self.__CaptureThread)
        #self.__thread.start()

        self.read("../media/cameraLeft.mov", "../media/cameraRight.mov")
        return True

    def Stop(self):
        """Stop the main thread."""
        if self.__isRunning is not True:
            return False

        self.__isRunning = False
        self.__thread.join(1000)
        return True

    def Clear(self):
        """Empty all internal parameters used for this class."""
        self.hasFundamentalMatrix = self.IsCalibrating = self.IsSaving = self.IsFrozen = False
        self.PointsQueue   = deque(maxlen=16)

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#

    def read(self, left_video_path, right_video_path):
        UMedia.load_videos([left_video_path, right_video_path], self.process)

    def process(self, images):
        leftImage = cv2.pyrDown(images[0])
        rightImage = cv2.pyrDown(images[1])

        leftCorners  = Configuration.Instance.Pattern.FindCorners(leftImage)
        rightCorners = Configuration.Instance.Pattern.FindCorners(rightImage)

        # Check if the calibration process is running.
        if self.IsCalibrating:
            # If both pattern have been recognized, start the calibration process.
            if leftCorners is not None and rightCorners is not None:
                self.__Calibrate(leftCorners, rightCorners)
            # Otherwise, stop the calibrations process.
            else:
                self.IsCalibrating = False

        # Check if the system is calibrated.
        elif Configuration.Instance.Calibration.IsCalibrated:
            # Estimate the depth map from two stereo images.
            self.__DepthMap(leftImage, rightImage)

        # Combine two stereo images in only one window.
        self.Image = self.__CombineImages(leftImage, rightImage, 0.5)
        cv2.imshow("Original", self.Image)

    def __CaptureThread(self):
        """Main thread of this system."""
        # Creates a window to show the original images.
        cv2.namedWindow("Original",  cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Original", self.mouseEvent)

        # Creates a window to show the stereo images.
        cv2.namedWindow("Stereo",  cv2.WINDOW_AUTOSIZE)

        # Creates a window to show the depth map.
        cv2.namedWindow("DepthMap", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("minDisparity", "DepthMap", 1, 32, self.__SetMinDisparity)
        cv2.createTrackbar("blockSize",    "DepthMap", 1,  5, self.__SetNothing)

        # Repetition statement for analyzing each captured image.
        while True:
    
            # Check if the fundamental matrix process is running.
            if not self.IsFrozen:

                # Grabs the next frame from capturing device.
                StereoCameras.Instance.Grab()
                # Decodes and returns the grabbed video frames.
                leftImage, rightImage = StereoCameras.Instance.Retrieve()

                # Find the pattern in the image.
                leftCorners  = Configuration.Instance.Pattern.FindCorners(leftImage)
                rightCorners = Configuration.Instance.Pattern.FindCorners(rightImage)

                # Check if the calibration process is running.
                if self.IsCalibrating:
                    # If both pattern have been recognized, start the calibration process.
                    if leftCorners is not None and rightCorners is not None:
                        self.__Calibrate(leftCorners, rightCorners)
                    # Otherwise, stop the calibrations process.
                    else:
                        self.IsCalibrating = False

                # Check if the system is calibrated.
                elif Configuration.Instance.Calibration.IsCalibrated:
                    # Estimate the depth map from two stereo images.
                    self.__DepthMap(leftImage, rightImage)

                # Combine two stereo images in only one window.
                self.Image = self.__CombineImages(leftImage, rightImage, 0.5)
                cv2.imshow("Original", self.Image)

            # Check what the user wants to do.
            inputKey = cv2.waitKey(1)
            # Esc or letter "q" key.
            if inputKey == 27 or inputKey == ord("q"):
                break
            # Space key.
            elif inputKey == 32:
                self.IsCalibrating = True
            # Letter "s" key.
            elif inputKey == ord("s"):
                self.IsSaving = True
            # Letter "f" key.
            elif inputKey == ord("f"):
                self.IsFrozen = not self.IsFrozen

        # Closes all video capturing devices.
        StereoCameras.Instance.Release()
        # Close all OpenCV windows.
        cv2.destroyAllWindows()

    def __FundamentalMatrix(self, point):
        # Check if the image is frozen.
        # SIGB: The user can frozen the input image presses "f" key.
        if self.IsFrozen:

            # Insert the new selected point in the queue.
            if self.__UpdateQueue(point):

                # Get all points selected by the user.
                points = np.asarray(self.PointsQueue, dtype=np.float32)

                # <000> Get the selected points from the left and right images.

                # <001> Estimate the Fundamental Matrix.

                # <002> Save the Fudamental Matrix in the F attribute of the CamerasParameters class.

                # Get each point from left image.
                # for point in left:

                    # <003> Estimate the epipolar line.

                    # <004> Define the initial and final points of the line.
                
                    # <005> Draws the epipolar line in the input image.

                # Get each point from right image.
                # for point in right:

                    # <006> Estimate the epipolar line.

                # Show the final result of this process to the user.
                cv2.imshow("Original", self.Image)

                # Update the fundamental matrix flag and release the system
                self.hasFundamentalMatrix = True

    def __Calibrate(self, leftCorners, rightCorners):
        """Calibrate the stereo camera for each new detected pattern."""
        # Get The outer vector contains as many elements as the number of the pattern views.
        objectPoints = Configuration.Instance.Pattern.CalculePattern()

        # <007> Insert the pattern detection results in three vectors.

        # Get the parameters used for calibrating each stereo camera.
        leftCorners  = Configuration.Instance.Pattern.LeftCorners
        rightCorners = Configuration.Instance.Pattern.RightCorners
        objectPoints = Configuration.Instance.Pattern.ObjectPoints

        # <008> Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

        # <008> Write the camera intrinsic and extrinsic parameters.

        # Calibrates the stereo camera.
        R, T = Configuration.Instance.Calibration.StereoCalibrate(leftCorners, rightCorners, objectPoints)

        # <015> Computes rectification transforms for each head of a calibrated stereo camera.

        # <016> Computes the undistortion and rectification transformation maps.

        # End the calibration process.
        self.IsCalibrating = False

        # Stop the system for 1 second, because the user will see the processed images.
        cv2.waitKey(1000)

    def __Epipolar(self, point):
        """Define the points used during the fundamental matrix process."""
        pass

    def __DepthMap(self, leftImage, rightImage):
        """Estimate the depth map from two stereo images."""
        # <017> Create the stereo image.

        # <018> Get the attributes for the block matching algorithm.
        # SIGB: minDisparity needs to be divisible by 16 and block size needs to be an odd number.

        # <019> Computing a stereo correspondence using the block matching algorithm.

        # Check if it is necessary to save the PLY file.
        if self.IsSaving:
            self.__SavePLY(disparity, leftStereo)

        # <020> Normalizes the disparity image for a valid output OpenCV image.

        # Shows the disparity image.
        cv2.imshow("DepthMap", disparity)

        # Combine two stereo images in only one window.
        stereo = self.__CombineImages(leftStereo, rightStereo, 0.5)
        cv2.imshow("Stereo", stereo)

    def __SavePLY(self, disparity, image):
        # Check if the system is calibrated.
        if Configuration.Instance.Calibration.IsCalibrated:
            # Get a 4x4 disparity-to-depth mapping matrix.
            Q = StereoCameras.Instance.Parameters.Q

            # Reprojects a disparity image to 3D space.
            points = cv2.reprojectImageTo3D(disparity, Q)

            # Creates a mask of the depth mapping matrix.
            mask = disparity > disparity.min()
            points = points[mask].reshape(-1, 3)

            # Defines the output numpy array.
            output = points

            # Save the output file.
            with open("OpenCV3D.ply", "w") as filename:
                filename.write(OpenCV3D.___plyHeader % dict(num = len(output)))
                np.savetxt(filename, output, "%f %f %f", newline="\n")            
            
            # End the PLY save process.
            self.IsSaving = False

    def __CombineImages(self, image1, image2, scale=1):
        """Combine two image in only one visualization."""
        # Define the final size.
        height, width = image1.shape[:2]
        width  = int(width  * scale)
        height = int(height * scale)

        # Define the new size to input images.
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        # Create the final image.
        image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        image[:height,      :width    ]  = image1
        image[:height, width:width * 2] = image2

        # Return the combine images.
        return image

    #----------------------------------------------------------------------#
    #                      Class Action Events Methods                     #
    #----------------------------------------------------------------------#
    def mouseEvent(self, event, x, y, flag, param):
        """This is an example of a calibration process using the mouse clicks."""
        # Starts the PLY save process.
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.hasFundamentalMatrix:
                self.__Epipolar((x, y))
            else:
                self.__FundamentalMatrix((x, y))

        # Reset all configuration variables.
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.Clear()
            Configuration.Instance.Clear()

        # Starts the calibration process.
        elif event == cv2.EVENT_RBUTTONUP:
            self.IsCalibrating = True

    def __UpdateQueue(self, point):
        """Insert a new point in the queue."""
        # Get the current queue size.
        size = len(self.PointsQueue)

        # Check if the queue is full.
        if size == self.PointsQueue.maxlen:
            return True

        # Defines the color used for draw the circle and the line.
        color = (0, 0, 255) if size % 2 == 0 else (255, 0, 0)

        # Draw a circle in the selected point.
        cv2.circle(self.Image, point, 3, color, thickness=-1)
        cv2.imshow("Original", self.Image)

        # Adjust the right click to correct position.
        if size % 2 != 0:
            point = (point[0] - 320, point[1])

        # It is necessary to update the selected point, because the systems shows a resized input image.
        # SIBG: You can use the original size, if you call __CombineImages() method with scale factor value 1.0.
        point = (point[0] * 2, point[1] * 2, 1)

        # Insert the new point in the queue.
        self.PointsQueue.append(point)

        # Check if the queue is full now.
        if size + 1 == self.PointsQueue.maxlen:
            return True

        # It is necessary to add more points.
        return False

    def __SetMinDisparity(self, value):
        """Masks the minDisparity variable."""
        if value == 0:
            cv2.setTrackbarPos("minDisparity", "DepthMap", int(1))

    def __SetNothing(self, value):
        """Standard mask."""
        pass

#----------------------------------------------------------------------#
#                             Main Methods                             #
#----------------------------------------------------------------------#
def main(argv):
    OpenCV3D().Start()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
