#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!-- File       : Augmented.py                                             -->
#<!-- Description: Class used for performing the augmented reality process  -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 18/04/2015                                               -->
#<!-- Change     : 18/04/2015 - Creation of these classes                   -->
#<!-- Review     : 18/04/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015041801 $'

########################################################################
import cv2
import numpy as np
import math
import Configuration
from Cameras.StereoCameras      import StereoCameras

########################################################################
class Augmented(object):
    """Calibration Class is used for performing the augmented reality process."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Cube(self):
        """Get an augmented cube."""
        return self.__cube

    @property
    def CoordinateSystem(self):
        """Get an augmented coordinate system."""
        return self.__coordinateSystem

    @property
    def __LightSource(self):
        return self.__lightSource

    #----------------------------------------------------------------------#
    #                      Augmented Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Augmented Class Constructor."""
        # Creates the augmented objects used by this class.
        self.__CreateObjects()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def PoseEstimation(self, objectPoints, corners, points, cameraMatrix, distCoeffs):
        """Define the pose estimation of the calibration pattern."""
        # <023> Find the rotation and translation vectors.
        # Reshape object points for solvePnP function
        # objectPoints = objectPoints.reshape(objectPoints.shape + (1,))

        retval, rvec, tvec = cv2.solvePnP(objectPoints, corners, cameraMatrix, distCoeffs)

        # Save the rotation and translation matrices as private attributes.
        self.__rotation    = cv2.Rodrigues(rvec)[0]
        self.__translation = tvec

        # <024> Project 3D points to image plane.
        # imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, points)
        imagePoints, jacobian = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)

        # Return the final result.
        return imagePoints

    def ApplyTexture(self, image, filename, points):
        """Applies a texture mapping over an augmented virtual object."""
        # Get the size of the analyzed image.
        h, w = image.shape[:2]

        # <031> Open the texture mapping image and get its size.
        textureMap = cv2.imread(filename)
        th, tw = textureMap.shape[:2]

        # Creates a mask with the same size of the input image.
        whiteMask = np.ones(textureMap.shape[:2], dtype=np.uint8) * 255
        blackMask = np.zeros(image.shape[:2], dtype=np.uint8)
        # cv2.imshow("mask", whiteMask)
        # TODO: FINISH

        # <032> Estimate the homography matrix between the texture mapping and the cube face.
        # srcPts = np.float32([[0, 0], [tw, 0], [0, th], [tw, th]])
        srcPts = np.float32([[0, th], [tw, th], [tw, 0], [0, 0]])


        points = points.reshape(4,2)

        # print "appText:"
        # print "----------------------------"
        # print srcPts
        # print "----------------------------"
        # print points
        # print "----------------------------"

        H, _ = cv2.findHomography(srcPts, points)

        # <033> Applies a perspective transformation to the texture mapping image.
        textureWarped = cv2.warpPerspective(textureMap, H, (w, h))
        whiteMask = cv2.warpPerspective(whiteMask, H, (w, h))


        # tw = cv2.cvtColor(textureWarped, cv2.COLOR_BGR2GRAY)
        # blackMask = cv2.cvtColor(blackMask, cv2.COLOR_GRAY2BGR)
        # whiteMask = cv2.cvtColor(whiteMask, cv2.COLOR_GRAY2BGR)
        tw = textureWarped
        bin = cv2.bitwise_or(blackMask, whiteMask)
        # return tw, bin

        # bin = cv2.bitwise_or(blackMask, whiteMask)

        # ttw = cv2.bitwise_and(bin, tw)

        # tmpImg = cv2.bitwise_or(image, cv2.cvtColor(ttw, cv2.COLOR_GRAY2BGR))

        # tmpImg = cv2.bitwise_or(image, tw)
        # tmpImg = cv2.bitwise_or(image, tw, mask = bin)
        # tmpImg2 = cv2.bitwise_or(tw, tmpImg)

        bin2 = cv2.bitwise_not(bin)


        tmpImg = cv2.bitwise_and(image, cv2.cvtColor(bin2, cv2.COLOR_GRAY2BGR))

        return tw, bin

        # return tmpImg, tw,

        # tmpImg2 = cv2.bitwise_or(tmpImg, tw)



        # cv2.imshow("whiteMask", whiteMask)
        # cv2.imshow("blackMask", blackMask)

        # cv2.imshow("tw", tw)
        # cv2.imshow("ttw", ttw)
        # cv2.imshow("bin", bin)

        # cv2.imshow("Masked", tmpImg)
        # cv2.imshow("Final", tmpImg2)



        # <034> Create a mask from the cube face using the texture mapping image.
        # return tw



    def ShadeFace(self, image, points, normals, projections, corners):
        shadeRes = 10
        h, w = image.shape[:2]

        square = np.array([[0, 0], [shadeRes-1, 0], [shadeRes-1, shadeRes-1], [0, shadeRes-1]])

        # tf = Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, TopFace, cameraMatrix, distCoeffs)
        objectPoints = Configuration.Configuration.Instance.Pattern.CalculatePattern()
        #
        # # <021> Prepares the external parameters.
        # if camera == CameraEnum.LEFT:
        cameraMatrix = StereoCameras.Instance.Parameters.CameraMatrix1
        distCoeffs = StereoCameras.Instance.Parameters.DistCoeffs1

        p = Configuration.Configuration.Instance.Augmented.PoseEstimation(objectPoints, corners, points, cameraMatrix, distCoeffs)

        # p = np.delete(points, 2, 1)
        square = square.astype(float)
        p = p.reshape((4,2))
        p = p.astype(float)
        projections = projections.reshape((4,2))

        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print p
        # print square
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # H, _ = cv2.findHomography(square, projections)
        H, _ = cv2.findHomography(square, p)
        Mr0, Mg0, Mb0= self.__CalculateShadeMatrix(image, shadeRes, points, normals)

        Mr = cv2.warpPerspective(Mr0, H, (w, h), flags=cv2.INTER_LINEAR)
        Mg = cv2.warpPerspective(Mg0, H, (w, h), flags=cv2.INTER_LINEAR)
        Mb = cv2.warpPerspective(Mb0, H, (w, h), flags=cv2.INTER_LINEAR)

        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        [r,g,b] = cv2.split(image2)

        whiteMask = np.copy(r)
        whiteMask[:,:]=[0]
        points_Proj2=[]
        points_Proj2.append([int(projections[0, 0]),int(projections[0, 1])])
        points_Proj2.append([int(projections[1, 0]),int(projections[1, 1])])
        points_Proj2.append([int(projections[2, 0]),int(projections[2, 1])])
        points_Proj2.append([int(projections[3, 0]),int(projections[3, 1])])

        cv2.fillConvexPoly(whiteMask, np.array(points_Proj2),(255,255,255))

        r[np.nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),r[np.nonzero(whiteMask>0)]*Mr[np.nonzero(whiteMask>0)])
        g[np.nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),g[np.nonzero(whiteMask>0)]*Mg[np.nonzero(whiteMask>0)])
        b[np.nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),b[np.nonzero(whiteMask>0)]*Mb[np.nonzero(whiteMask>0)])

        image2=cv2.merge((r,g,b))
        image=cv2.cvtColor(image2, cv2.COLOR_RGB2BGR, image)

    def __CalculateShadeMatrix(self, image, size, points, normals):
        # create the empty lighting texture
        shade = np.zeros((size, size, 3))

        # Ambient Light
        IA = np.array([5.0, 5.0, 5.0])

        # Point Light
        IP = np.array([5.0, 5.0, 5.0])

        ka = np.array([0.2, 0.2, 0.2])  # ambient
        kd = np.array([0.3, 0.3, 0.3])  # diffuse
        ks = np.array([0.7, 0.7, 0.7])  # specular

        # used for specular
        alpha = 100

        faceNormal, center, angle = self.GetFaceNormal(points)
        cameraCenter = -np.dot(self.__rotation.T, self.__translation).T

        viewVector = cameraCenter - center
        viewVector = viewVector / np.linalg.norm(viewVector)

        # read light position from global (mouse callback) if set
        if self.__LightSource == None:
            lightPos = cameraCenter
            self.__lightSource = lightPos
        else:
            lightPos = self.__LightSource

        lightIncidenceVector = lightPos - center
        lightIncidenceVector = lightIncidenceVector / np.linalg.norm(lightIncidenceVector)
        lightIncidenceVector = lightIncidenceVector.reshape((3,))
        viewVector = viewVector.reshape((3,))

        print "----------------------------------"
        print lightIncidenceVector
        print "----------------------------------"
        print center
        print "----------------------------------"
        print cameraCenter

        # face reflection vector
        lightReflectionVector = 2 * np.dot(lightIncidenceVector, faceNormal) * faceNormal - lightIncidenceVector

        for y, row in enumerate(shade):
            for x, value in enumerate(row):
                interpolatedFaceNormal = self.__BilinearInterpolation(size, x, y, normals, True)
                interpolatedFaceNormal = np.array(interpolatedFaceNormal)

                # light
                light = max(np.dot(interpolatedFaceNormal, lightIncidenceVector), 0)

                # specular
                lightReflectionVector = 2 * np.dot(lightIncidenceVector, interpolatedFaceNormal) * interpolatedFaceNormal - lightIncidenceVector
                spec = pow(max(0, np.dot(lightReflectionVector, viewVector)), alpha)

                # put it all together
                shade[y][x] = IA * ka + IP * kd * light + IP * ks * spec

        # return all three channels
        return shade[:, :, 0], shade[:, :, 1], shade[:, :, 2]

    def GetFaceNormal(self, points):
        """Get some information of a correspoding cube face."""
        # Estimate the normal vector of the corresponding cube face.
        A = np.subtract([points[1, 0], points[1, 1], points[1, 2]], [points[0, 0], points[0, 1], points[0, 2]])
        B = np.subtract([points[2, 0], points[2, 1], points[2, 2]], [points[0, 0], points[0, 1], points[0, 2]])
        normal = np.cross(A, B)
        normal = normal / np.linalg.norm(normal)

        # Calculate the midpoint of the corresponding cube face.
        center = np.mean(points, axis=0)

        # Estimate the vector from camera center to cube face center.
        cameraCenter = -np.dot(self.__rotation.T, self.__translation).T
        C = np.subtract(cameraCenter, center)
        C = C / np.linalg.norm(C)

        # Calculate the angle of the normal vector.
        angle = np.arccos(np.dot(C, normal)) * (180 / np.pi)

        # Return the results.
        return normal, center, angle

    def CalculateFaceCornerNormals(self, top, right, left, up, down):
        cubeCornerNormals = self.GetNormalsInCubeCorners(top, right, left, up, down)

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]])
        t = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[6, 2, 3, 7], [6, 2, 3, 7], [6, 2, 3, 7]])
        r = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[4, 0, 1, 5], [4, 0, 1, 5], [4, 0, 1, 5]])
        l = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[5, 1, 2, 6], [5, 1, 2, 6], [5, 1, 2, 6]])
        u = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[7, 3, 0, 4], [7, 3, 0, 4], [7, 3, 0, 4]])
        d = cubeCornerNormals[i, j].T

        return t, r, l, u, d

    def GetNormalsInCubeCorners(self, top, right, left, up, down):
        points = []

        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(left)[0]  - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(right)[0] - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(right)[0] - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(left)[0]  - self.GetFaceNormal(top)[0]) / 3)

        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(left)[0]  + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(right)[0] + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(right)[0] + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(left)[0]  + self.GetFaceNormal(top)[0]) / 3)

        return np.array(points).T

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __CreateObjects(self):
        """Defines the points of the augmented objects based on calibration patterns."""
        # Creates an augmented cube.
        self.__cube = np.float32([[3, 1,  0], [3, 4,  0], [6, 4,  0], [6, 1,  0],
                                  [3, 1, -3], [3, 4, -3], [6, 4, -3], [6, 1, -3]])
        # Creates the coordinate system.
        self.__coordinateSystem = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)

        self.__lightSource = None

    def __BilinearInterpolation(self, size, i, j, points, isNormalized):
        x1 = 0
        y1 = 0
        x2 = size
        y2 = size

        Q11 = points[0, 0]
        Q21 = points[1, 0]
        Q22 = points[2, 0]
        Q12 = points[3, 0]
        X = (1.0 / ((x2-x1) * (y1-y2))) * (Q12 * (x2-i) * (y1-j) + Q22 * (i-x1) * (y1-j) + Q11 * (x2-i) * (j-y2) + Q21 * (i-x1) * (j-y2))

        Q11 = points[0, 1]
        Q21 = points[1, 1]
        Q22 = points[2, 1]
        Q12 = points[3, 1]
        Y = (1.0 / ((x2-x1) * (y1-y2))) * (Q12 * (x2-i) * (y1-j) + Q22 * (i-x1) * (y1-j) + Q11 * (x2-i) * (j-y2) + Q21 * (i-x1) * (j-y2))

        Q11 = points[0, 2]
        Q21 = points[1, 2]
        Q22 = points[2, 2]
        Q12 = points[3, 2]
        Z = (1.0 / ((x2-x1) * (y1-y2))) * (Q12 * (x2-i) * (y1-j) + Q22 * (i-x1) * (y1-j) + Q11 * (x2-i) * (j-y2) + Q21 * (i-x1) * (j-y2))

        if isNormalized:
            normal = (X, Y, Z) / np.linalg.norm((X, Y, Z))
            X = normal[0]
            Y = normal[1]
            Z = normal[2]

        return X, Y, Z

    def __Shade(self, x, n, lightInWorld):
        n = np.array(n)
        shade = np.array(self.__Ambient(x).T)[0] + np.array(self.__Diffuse(x, n, lightInWorld)).T + np.array(self.__Glossy(x,n, lightInWorld)).T

        return shade

    def __Ambient(self, x):
        # Ambient light: IA = [IaR,IaG,IaB].
        IA = np.matrix([5.0, 5.0, 5.0]).T
        # Material properties: Ka = [kaR; kaG; kaB].
        ka = np.matrix([0.2, 0.2, 0.2]).T

        return np.multiply(IA, ka)

    def __Diffuse(self, x,n,light):
        #Point light IA=[IpR,IpG,IpB]
        IP = np.matrix([5.0, 5.0, 5.0]).T
        kd= np.matrix([0.3, 0.3, 0.3]).T

        l = np.subtract (light,x)
        l = np.array(l).T/ np.linalg.norm(np.array(l))

        a = np.dot(n, l)

        a = np.maximum(a, 0)
        a=np.power(a,5)# This is not in the actual formula and it is only for increasing the effect of the viewing angle
    #    r=dist(x, l)
        d= np.multiply(IP, kd) * a

        return d

    def __Glossy(self, x,n, light):
        #Point light IA=[IpR,IpG,IpB]
        IP = np.matrix([5.0, 5.0, 5.0]).T
        ks=np.matrix([0.7, 0.7, 0.7]).T
        alpha = 100

        cameraCenter = np.dot(-self.__rotation.T, self.__translation).T[0]
        v=np.subtract(cameraCenter,x)
        v=np.array(v).T/ np.linalg.norm(np.array(v))


        r= np.subtract(2*n*np.dot(n,v),v)

        rv=max(np.dot(r,v),0)
    #    print "rv:", rv
    #    print "angle",math.acos(rv) * 57.2957795


        cc=np.multiply(IP, ks) * pow(rv, alpha)
    #    print "cc",cc
        return cc
