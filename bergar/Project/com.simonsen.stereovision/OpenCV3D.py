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
import cv2
import sys
import numpy as np

from pylab                      import *
from collections                import deque
from threading                  import Thread

from Cameras.CameraEnum         import CameraEnum
from Cameras.StereoCameras      import StereoCameras
from Processing.Configuration   import Configuration
from Cameras                    import CamerasParameters
from Processing.Calibration     import Calibration
from Processing.CalibrationEnum import CalibrationEnum
from Processing.Augmented       import Augmented

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

    @property
    def IsDrawing(self):
        """Check if the system is drawing some object."""
        return self.__isDrawing

    @IsDrawing.setter
    def IsDrawing(self, value):
        """Set that the system will draw objects."""
        self.__isDrawing = value

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
        self.__isRunning = True
        self.__thread = Thread(target=self.__CaptureThread)
        self.__thread.start()
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
        self.hasFundamentalMatrix = self.IsCalibrating = self.IsSaving = self.IsFrozen = self.IsDrawing = False
        self.PointsQueue   = deque(maxlen=16)

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
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
                leftCorners  = Configuration.Instance.Pattern.FindCorners(leftImage,  not self.IsDrawing)
                rightCorners = Configuration.Instance.Pattern.FindCorners(rightImage, not self.IsDrawing)

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
                    # Check if the system is drawing some object.
                    if self.IsDrawing:
                        # If both pattern have been recognized, start the calibration process.
                        if leftCorners is not None and rightCorners is not None:
                            self.__Augmentation(leftCorners,  leftImage)
                            # self.__Augmentation(rightCorners, rightImage, True)
                            # TODO: Uncomment or delete
                    # Otherwise, estimates the depth map from two stereo images.
                    else:
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
            # Letter "d" key.
            elif inputKey == ord("d"):
                self.IsDrawing = not self.IsDrawing

        # Closes all video capturing devices.
        StereoCameras.Instance.Release()
        # while True:
        #
        #     # Check if the fundamental matrix process is running.
        #     if not self.IsFrozen:
        #
        #         # Grabs the next frame from capturing device.
        #         StereoCameras.Instance.Grab()
        #         # Decodes and returns the grabbed video frames.
        #         leftImage, rightImage = StereoCameras.Instance.Retrieve()
        #
        #         # Find the pattern in the image.
        #         leftCorners  = Configuration.Instance.Pattern.FindCorners(leftImage)
        #         rightCorners = Configuration.Instance.Pattern.FindCorners(rightImage)
        #
        #         # Check if the calibration process is running.
        #         if self.IsCalibrating:
        #             # If both pattern have been recognized, start the calibration process.
        #             if leftCorners is not None and rightCorners is not None:
        #                 self.__Calibrate(leftCorners, rightCorners)
        #             # Otherwise, stop the calibrations process.
        #             else:
        #                 self.IsCalibrating = False
        #
        #         # Check if the system is calibrated.
        #         elif Configuration.Instance.Calibration.IsCalibrated:
        #             # Estimate the depth map from two stereo images.
        #             self.__DepthMap(leftImage, rightImage)
        #
        #         # Combine two stereo images in only one window.
        #         self.Image = self.__CombineImages(leftImage, rightImage, 0.5)
        #         # self.Image = self.__CombineImages(leftImage, rightImage, 1.0) # TODO: Change back
        #         cv2.imshow("Original", self.Image)
        #
        #     # Check what the user wants to do.
        #     inputKey = cv2.waitKey(1)
        #     # Esc or letter "q" key.
        #     if inputKey == 27 or inputKey == ord("q"):
        #         break
        #     # Space key.
        #     elif inputKey == 32:
        #         self.IsCalibrating = True
        #     # Letter "s" key.
        #     elif inputKey == ord("s"):
        #         self.IsSaving = True
        #     # Letter "f" key.
        #     elif inputKey == ord("f"):
        #         self.IsFrozen = not self.IsFrozen
        #
        # # Closes all video capturing devices.
        # StereoCameras.Instance.Release()
        # Close all OpenCV windows.
        cv2.destroyAllWindows()

    def estimateFundamental(self, x1, x2):
        n = x1.shape[1]
        if x2.shape[1] != n:
            raise ValueError("Number of points do not match.")

        # Build matrix for equation
        A = np.zeros((n, 9))
        for i in range(n):
            A[i] = [
                x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[3, i] * x2[2, i],
                ]

        # Compute linear least square solution
        U, S, V = linalg.svd(A)
        F = V[-1].reshape(3, 3)

        # Constrain F
        # Make rank 2 by zeroing out last singular value
        U, S, V = linalg.svd(F)
        S[2] = 0
        F = dot(U, dot(diag(S), V))

        return F

    def getSelectedPoints(self, points):
        print points
        x1 = points[::2]
        x2 = points[1::2]
        print "---------------------------"
        print x1
        print "---------------------------"
        print x2
        return x1, x2

    def computeEpipole(self, F):
        U, S, V = linalg.svd(F)
        e = V[-1]
        return e / e[2]

    def getHardCodedPoints2(self):
        p = array([
            [4.08000000e+02, 2.18000000e+02],
            [9.56000000e+02, 2.92000000e+02],
            [3.66000000e+02, 6.84000000e+02],
            [9.00000000e+02, 7.58000000e+02],
            [1.05000000e+03, 3.22000000e+02],
            [1.60400000e+03, 4.10000000e+02],
            [1.09200000e+03, 6.92000000e+02],
            [1.64400000e+03, 7.82000000e+02],
            [3.62000000e+02, 1.82000000e+02],
            [9.10000000e+02, 2.50000000e+02],
            [8.98000000e+02, 8.40000000e+01],
            [1.45200000e+03, 1.72000000e+02],
            [1.07400000e+03, 1.34000000e+02],
            [1.63200000e+03, 2.20000000e+02],
            [8.86000000e+02, 2.82000000e+02],
            [1.44600000e+03, 3.66000000e+02]
        ])
        return p
    def getHardCodedPoints(self):
        p = array([
            [  4.08000000e+02, 2.18000000e+02, 1.00000000e+00],
            [  9.56000000e+02, 2.92000000e+02, 1.00000000e+00],
            [  3.66000000e+02, 6.84000000e+02, 1.00000000e+00],
            [  9.00000000e+02, 7.58000000e+02, 1.00000000e+00],
            [  1.05000000e+03, 3.22000000e+02, 1.00000000e+00],
            [  1.60400000e+03, 4.10000000e+02, 1.00000000e+00],
            [  1.09200000e+03, 6.92000000e+02, 1.00000000e+00],
            [  1.64400000e+03, 7.82000000e+02, 1.00000000e+00],
            [  3.62000000e+02, 1.82000000e+02, 1.00000000e+00],
            [  9.10000000e+02, 2.50000000e+02, 1.00000000e+00],
            [  8.98000000e+02, 8.40000000e+01, 1.00000000e+00],
            [  1.45200000e+03, 1.72000000e+02, 1.00000000e+00],
            [  1.07400000e+03, 1.34000000e+02, 1.00000000e+00],
            [  1.63200000e+03, 2.20000000e+02, 1.00000000e+00],
            [  8.86000000e+02, 2.82000000e+02, 1.00000000e+00],
            [  1.44600000e+03, 3.66000000e+02, 1.00000000e+00]
        ])

        return p

    def plotEpipolarLine(self, im, F, x, epipole=None, showEpipole=True):
        m,n = im.shape[:2]
        line = dot(F, x)

        t = linspace(0, n, 100)
        lt = array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

        # take only line points inside the image
        ndx = (lt >= 0) & (lt < m)
        plot(t[ndx], lt[ndx], linewidth=2)

        if showEpipole:
            if epipole is None:
                epipole = self.computeEpipole(F)
            # plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')
        # return epipole[0] / epipole[2], epipole[1] / epipole[2]
        return t[ndx], lt[ndx]


    def __FundamentalMatrix(self, point):
        # Check if the image is frozen.
        # SIGB: The user can frozen the input image presses "f" key.
        if self.IsFrozen:

            # Insert the new selected point in the queue.
            if self.__UpdateQueue(point):

                # Get all points selected by the user.
                points = np.asarray(self.PointsQueue, dtype=np.float32)

                # Hardcoded points
                # TODO: Remove when done
                points = self.getHardCodedPoints()

                # <000> Get the selected points from the left and right images.
                left, right = self.getSelectedPoints(points)

                # TODO: Remove ?
                x1 = np.float32(left)
                x2 = np.float32(right)
                x1 = np.delete(x1, 2, 1)
                x2 = np.delete(x2, 2, 1)


                # <001> Estimate the Fundamental Matrix.
                # F = self.estimateFundamental(x1, x2)
                F, mask = cv2.findFundamentalMat(x1, x2, cv2.cv.CV_FM_8POINT)

                # <002> Save the Fundamental Matrix in the F attribute of the CamerasParameters class.
                StereoCameras.Instance.Parameters.F = F
                # CamerasParameters.CamerasParameters.F = F

                # self.plotEpipolarLine(self.Image, F, [814, 148, 1])

                # le = self.computeEpipole(F.T)
                # re = self.computeEpipole(F)
                #
                # xl = le[0]
                # yl = le[1]
                # xl = int(xl)
                # yl = int(yl)
                #
                # xr = re[0]
                # yr = re[1]
                # xr = int(xr)
                # yr = int(yr)q
                #
                # cv2.circle(self.Image, (xl, yl), 600, (255, 255, 0))
                # cv2.circle(self.Image, (xr, yr), 600, (255, 0, 255))
                # cv2.circle(self.Image, (814, 148), 3, (0, 0, 255))

                e = self.computeEpipole(F)

                # print "-------- PRE"
                # print left
                #
                # lp = array(None)

                # for i in range(8):
                #     lp.append(self.plotEpipolarLine(self.Image, F, left[i], e, True))
                # lp.append([1, 100])
                # lp.append([100, 200])
                # lp.append([200, 250])
                # lp.append([250, 10])
                # lp.append([10, 99])
                # cv2.cv.ComputeCorrespondEpilines(left, self.Image, F, lp)
                #
                # print "LINE_POINTS"
                # print lp
                #
                # for i in range(len(lp)):
                #     if i > 0:
                #         cv2.line(self.Image, (int(lp[i-1][0]), int(lp[i-1][1])), (int(lp[i][0]), int(lp[i][1])), (255, 255, 0), 3)


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
        objectPoints = Configuration.Instance.Pattern.CalculatePattern()

        # <007> Insert the pattern detection results in three vectors.
        Configuration.Instance.Pattern.LeftCorners.append(leftCorners)
        Configuration.Instance.Pattern.RightCorners.append(rightCorners)
        Configuration.Instance.Pattern.ObjectPoints.append(objectPoints)


        # Get the parameters used for calibrating each stereo camera.
        leftCorners  = Configuration.Instance.Pattern.LeftCorners
        rightCorners = Configuration.Instance.Pattern.RightCorners
        objectPoints = Configuration.Instance.Pattern.ObjectPoints

        # <008> Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        calibration = Calibration()
        cameraMatrixLeft, distCoeffsLeft = calibration.CalibrateCamera(leftCorners, objectPoints, CameraEnum.LEFT)
        cameraMatrixRight, distCoeffsRight = calibration.CalibrateCamera(rightCorners, objectPoints, CameraEnum.RIGHT)

        # <008> Write the camera intrinsic and extrinsic parameters.
        # camParam = CamerasParameters.CamerasParameters()
        StereoCameras.Instance.Parameters.CameraMatrix1 = cameraMatrixLeft
        StereoCameras.Instance.Parameters.CameraMatrix2 = cameraMatrixRight
        StereoCameras.Instance.Parameters.DistCoeffs1 = distCoeffsLeft
        StereoCameras.Instance.Parameters.DistCoeffs2 = distCoeffsRight

        # Calibrates the stereo camera.
        R, T = Configuration.Instance.Calibration.StereoCalibrate(leftCorners, rightCorners, objectPoints)

        # <015> Computes rectification transforms for each head of a calibrated stereo camera.
        Configuration.Instance.Calibration.StereoRectify(R, T)

        # <016> Computes the undistortion and rectification transformation maps.
        Configuration.Instance.Calibration.UndistortRectifyMap()

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
        # TODO: Uncomment
        # if self.IsSaving:
        #     self.__SavePLY(disparity, leftStereo)

        # <020> Normalizes the disparity image for a valid output OpenCV image.

        # Shows the disparity image.
        # TODO: Uncomment
        # cv2.imshow("DepthMap", disparity)

        # Combine two stereo images in only one window.
        # stereo = self.__CombineImages(leftStereo, rightStereo, 0.5)
        stereo = self.__CombineImages(leftImage, rightImage, 0.5)
        cv2.imshow("Stereo", stereo)

    def __Augmentation(self, corners, image, camera=CameraEnum.LEFT):
        """Draws some augmentated object in the input image."""
        # Get The outer vector contains as many elements as the number of the pattern views.
        objectPoints = Configuration.Instance.Pattern.CalculatePattern()

        # <021> Prepares the external parameters.
        cameraMatrix = StereoCameras.Instance.Parameters.CameraMatrix2
        distCoeffs = StereoCameras.Instance.Parameters.DistCoeffs2

        # <022> Get the points of the coordinate system.
        points = Configuration.Instance.Augmented.CoordinateSystem


        # Defines the pose estimation of the coordinate system.
        points = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, points, cameraMatrix, distCoeffs)

        # <025> Draws the coordinate system over the chessboard pattern.
        # print "----------------------------"
        # print "Points:"
        # print points.shape
        # print points
        # print "----------------------------"

        # r = points[0]
        # p = r[0]
        # x = int(p[0])
        # y = int(p[1])
        # print x
        # print y
        # cv2.circle(image, (x, y), 10, (255, 0, 0), 3)
        # cv2.imshow("lines", image)

        r = points[0]
        s = r[0]
        x = int(s[0])
        y = int(s[1])
        cv2.circle(image, (x, y), 10, (255, 0, 0), 3)

        # tmpx = 0
        # tmpy = 0
        # for i in range(points.shape[0]):
        #     for p in points[i]:
        #         x = int(p[0])
        #         y = int(p[1])
        #         cv2.circle(image, (x, y), 10, (255, 0, 0), 3)
                # if (tmpx != 0) and (tmpy != 0):
                #     cv2.line(image, (tmpx, tmpy), (x, y), (255, 255, 0), 3)
                #     print "line (%d, %d) - (%d, %d)" % (tmpx, tmpy, x, y)
                # tmpx = x
                # tmpy = y
        # cv2.imshow("lines", image)
                # print "------------------"
                # print p[0]
                # print "------------------"
                # print p[1]
                # print "------------------"


        # <026> Get the points of the cube.
        cube = Configuration.Instance.Augmented.Cube

        # <027> Defines the pose estimation of the cube.
        cube = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, cube, cameraMatrix, distCoeffs)

        # <028> Draws ground floor in green color.
        # SIGB: Uses the last four points to do this.
        x1 = cube[50][0][0]
        y1 = cube[50][0][1]
        x2 = cube[51][0][0]
        y2 = cube[51][0][1]
        x3 = cube[52][0][0]
        y3 = cube[52][0][1]
        x4 = cube[53][0][0]
        y4 = cube[53][0][1]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,255), 2)
        cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), (255,0,255), 2)
        cv2.line(image, (int(x4), int(y4)), (int(x4), int(y4)), (255,0,255), 2)
        cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), (255,0,255), 2)

        # cv2.imshow("lines", image)
        # print x
        # print y
        # print "11111111111111111111111111"

        # <029> Draw pillars in blue color.
        # SIGB: Uses the intersections between the first four points and the last four points.

        # <030> Draw top layer in red color.
        # SIGB: Uses the first four points to do this.

        # Check if it is necessary to apply a texture mapping.
        if camera == CameraEnum.LEFT:
            return

        # Define each correspoding cube face.
        cube = Configuration.Instance.Augmented.Cube
        TopFace   = cube[4:]
        UpFace    = np.vstack([cube[5], cube[1:3], cube[6]])
        DownFace  = np.vstack([cube[7], cube[3],   cube[0], cube[4]])
        LeftFace  = np.vstack([cube[4], cube[0:2], cube[5]])
        RightFace = np.vstack([cube[6], cube[2:4], cube[7]])

        # Threshould used for selecting which cube faces will be drawn.
        threshold = 88

        # <035> Applies the texture mapping over all cube sides.

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
