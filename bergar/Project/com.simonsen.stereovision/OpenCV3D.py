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
from Processing                 import Utils
from Settings                   import Constant as C

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

                # leftImage, rightImage = StereoCameras.Instance.Retrieve()
                # TODO: Uncomment

                leftImage = Utils.get_frame_from_video(C.VIDEO_LEFT_4, 30)
                rightImage = Utils.get_frame_from_video(C.VIDEO_RIGHT_4, 30)

                # Find the pattern in the image.
                leftCorners  = Configuration.Instance.Pattern.FindCorners(leftImage,  not self.IsDrawing)
                rightCorners = Configuration.Instance.Pattern.FindCorners(rightImage, not self.IsDrawing)

                # Check if the calibration process is running.
                if self.IsCalibrating:
                    print "Start calibration"
                    # If both pattern have been recognized, start the calibration process.
                    if leftCorners is not None and rightCorners is not None:
                        self.__Calibrate(leftCorners, rightCorners)
                    # Otherwise, stop the calibrations process.
                    else:
                        self.IsCalibrating = False
                    print "Done Calibrating"

                # Check if the system is calibrated.
                elif Configuration.Instance.Calibration.IsCalibrated:
                    # Check if the system is drawing some object.
                    if self.IsDrawing:
                        print "Start drawing"
                        if leftCorners is not None and rightCorners is not None:
                            self.__Augmentation(leftCorners,  leftImage)
                            # self.__Augmentation(rightCorners, rightImage, CameraEnum.RIGHT)
                            # TODO: Uncomment or delete
                        print "Done drawing"
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
            # elif inputKey == ord("j"):
            #     lImg = Utils.get_frame_from_video(C.VIDEO_LEFT_3, 25)
            #     rImg = Utils.get_frame_from_video(C.VIDEO_RIGHT_3, 25)
            #     combImg = self.__CombineImages(lImg, rImg, 0.5)
            #     cv2.imshow("TESTING", combImg)

        # Closes all video capturing devices.
        StereoCameras.Instance.Release()
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
        return x1, x2

    def computeEpipole(self, F):
        U, S, V = linalg.svd(F)
        e = V[-1]
        return e / e[2]

    def getHardCodedPoints(self):
        p = array([
            [3.64000000e+02, 1.80000000e+02, 1.00000000e+00],
            [9.14000000e+02, 2.52000000e+02, 1.00000000e+00],
            [3.64000000e+02, 6.82000000e+02, 1.00000000e+00],
            [8.96000000e+02, 7.60000000e+02, 1.00000000e+00],
            [1.09800000e+03, 6.96000000e+02, 1.00000000e+00],
            [1.64600000e+03, 7.82000000e+02, 1.00000000e+00],
            [2.25000000e+03, 4.06000000e+02, 1.00000000e+00],
            [4.10000000e+02, 3.20000000e+02, 1.00000000e+00],
            [6.92000000e+02, 4.00000000e+01, 1.00000000e+00],
            [1.24400000e+03, 1.18000000e+02, 1.00000000e+00],
            [4.12000000e+02, 2.22000000e+02, 1.00000000e+00],
            [9.62000000e+02, 2.90000000e+02, 1.00000000e+00],
            [8.96000000e+02, 8.40000000e+01, 1.00000000e+00],
            [1.44800000e+03, 1.72000000e+02, 1.00000000e+00],
            [8.84000000e+02, 2.84000000e+02, 1.00000000e+00],
            [1.43800000e+03, 3.68000000e+02, 1.00000000e+00]
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

                points = self.getHardCodedPoints()

                # <000> Get the selected points from the left and right images.
                left, right = self.getSelectedPoints(points)

                left = np.array(left, dtype = np.float32)
                right = np.array(right, dtype = np.float32)


                # TODO: Remove ?
                left = np.float32(left)
                right = np.float32(right)
                left = np.delete(left, 2, 1)
                right = np.delete(right, 2, 1)


                # <001> Estimate the Fundamental Matrix.
                F, mask = cv2.findFundamentalMat(left, right)

                # <002> Save the Fundamental Matrix in the F attribute of the CamerasParameters class.
                CamerasParameters.CamerasParameters.F = F

                # self.build_epipolar_lines(left, F, False)
                # self.build_epipolar_lines(right, F, True)

                # Update the fundamental matrix flag and release the system
                e = self.computeEpipole(F)

                # Update the fundamental matrix flag and release the system
                self.hasFundamentalMatrix = True

    def build_epipolar_lines(self, points, fundamental_matrix, is_right, show_lines=True):
        lines = cv2.computeCorrespondEpilines(points, 2 if is_right else 1, fundamental_matrix)
        lines = lines.reshape(-1, 3)

        if show_lines:
            self.draw_lines(self.Image, lines, points, is_right)

    def draw_lines(self, img, lines, points, is_right):
        height, width, layers = img.shape

        color = (0, 0, 255) if not is_right else (255, 0, 0)
        x_gap_point = 0 if not is_right else width / 2
        x_gap_line = 0 if is_right else width / 2

        for height, row in zip(lines, points):
            x_start, y_start = map(int, [0, -height[2]/height[1]])
            x_end, y_end = map(int, [width/2, -(height[2]+height[0]*(width/2))/height[1]])

            cv2.line(img, (x_start + x_gap_line, y_start), (x_end + x_gap_line, y_end), color, 1)
            cv2.circle(img, (int(row[0] + x_gap_point), int(row[1])), 3, color)

        return img

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
        if camera == CameraEnum.LEFT:
            cameraMatrix = StereoCameras.Instance.Parameters.CameraMatrix1
            distCoeffs = StereoCameras.Instance.Parameters.DistCoeffs1
        else:
            cameraMatrix = StereoCameras.Instance.Parameters.CameraMatrix2
            distCoeffs = StereoCameras.Instance.Parameters.DistCoeffs2

        # <022> Get the points of the coordinate system.
        points = Configuration.Instance.Augmented.CoordinateSystem


        # Defines the pose estimation of the coordinate system.
        coordEstimation = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, points, cameraMatrix, distCoeffs)

        # <025> Draws the coordinate system over the chessboard pattern.
        corner = tuple(corners[0].ravel())
        cv2.line(image, corner, tuple(coordEstimation[0].ravel()), (255,0,0), 5)
        cv2.line(image, corner, tuple(coordEstimation[1].ravel()), (0,255,0), 5)
        cv2.line(image, corner, tuple(coordEstimation[2].ravel()), (0,0,255), 5)

        # <026> Get the points of the cube.
        cube = Configuration.Instance.Augmented.Cube
        # <027> Defines the pose estimation of the cube.
        cubeEstimation = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, cube, cameraMatrix, distCoeffs)
        cubeEstimation = np.int32(cubeEstimation).reshape(-1,2)

        # <028> Draws ground floor in green color.
        # SIGB: Uses the last four points to do this.
        cv2.drawContours(image, [cubeEstimation[:4]],-1,(0,255,0),-3)

        # <029> Draw pillars in blue color.
        # SIGB: Uses the intersections between the first four points and the last four points.
        for i,j in zip(range(4),range(4,8)):
            cv2.line(image, tuple(cubeEstimation[i]), tuple(cubeEstimation[j]),(255),3)

        # <030> Draw top layer in red color.
        # SIGB: Uses the first four points to do this.
        cv2.drawContours(image, [cubeEstimation[4:]],-1,(0,0,255),3)

        # Check if it is necessary to apply a texture mapping.
        if camera == CameraEnum.RIGHT:
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

        uf = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, UpFace, cameraMatrix, distCoeffs)
        rf = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, RightFace, cameraMatrix, distCoeffs)
        tf = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, TopFace, cameraMatrix, distCoeffs)
        lf = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, LeftFace, cameraMatrix, distCoeffs)
        df = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, DownFace, cameraMatrix, distCoeffs)

        # print "oooooooooooooooooooooooooooooooooooooooo"
        # print uf.shape
        # print uf
        # print "oooooooooooooooooooooooooooooooooooooooo"
        # print UpFace.shape
        # print UpFace
        # print "oooooooooooooooooooooooooooooooooooooooo"

        # <035> Applies the texture mapping over all cube sides.
        # Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_UP, uf)
        # Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_RIGHT, rf)
        # Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_UP, UpFace)
        Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_TOP, tf)
        # Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_LEFT, lf)
        # Configuration.Instance.Augmented.ApplyTexture(image, C.TEXTURE_DOWN, df)


    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        print tuple(imgpts[0].ravel())
        print tuple(imgpts[1].ravel())
        print tuple(imgpts[2].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

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
