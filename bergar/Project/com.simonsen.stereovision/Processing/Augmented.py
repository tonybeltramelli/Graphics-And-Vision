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
        H, _ = cv2.findHomography(srcPts, points)
        # H, _ = cv2.findHomography(points, srcPts)
        # H2, _ = cv2.findHomography(srcPts, points)
        # H, _ = cv2.findHomography(points, srcPts)

        print points
        print "----"
        print srcPts


        # cv2.drawContours(blackMask, [p2],-1,(0,255,0),-3)

        # print H

        # <033> Applies a perspective transformation to the texture mapping image.
        textureWarped = cv2.warpPerspective(textureMap, H, (w, h))

        wFirst = 0.1
        wSecond = 0.9
        gamma = 9
        M = cv2.addWeighted(image, wFirst, textureWarped, wSecond, gamma)


        cv2.imshow("testsetst", textureWarped)

        # print textureWarped.shape

        # blackMask = np.zeros(image.shape[:2], dtype=np.uint8)
        # whiteMask = np.ones(textureWarped.shape[:2], dtype=np.uint8) * 255
        # whiteMask = cv2.resize(whiteMask, (w, h))



        # print whiteMask

        # cv2.imshow("Whitemask", whiteMask)
        # cv2.imshow("blackMask", blackMask)

        # print blackMask.shape
        #
        # print whiteMask.shape

        # tmp = cv2.bitwise_and(blackMask, whiteMask)
        # cv2.imshow("bit and", tmp)



        # cv2 . resize ( newbackGround , (h , w ) )


        # mask = np.zeros(image.shape[:2], dtype = "uint8")
        # (cX, cY) = (image.shape[1] / 2, image.shape[0] / 2)
        # cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
        # cv2.imshow("mask", mask)

        # <034> Create a mask from the cube face using the texture mapping image.
        # blackMask = np.zeros(image.shape[:2], dtype=np.uint8)
        # whiteMask = np.ones(textureMap.shape[:2], dtype=np.uint8) * 255
        #
        # print blackMask.shape
        # print whiteMask.shape
        #
        # for i in range(whiteMask.shape[0]):
        #     for j in range(whiteMask.shape[1]):
        #         blackMask[i][j] = whiteMask[i][j]


        # cv2.bitwise_and(blackMask, whiteMask)

        # cv2.imshow("whiteMask", whiteMask)
        # cv2.imshow("blackMask", blackMask)



    def ShadeFace(self, image, points, normals, projections):
        shadeRes = 10
        h, w = image.shape[:2]

        square = np.array([[0, 0], [shadeRes-1, 0], [shadeRes-1, shadeRes-1], [0, shadeRes-1]])

        H, _ = cv2.findHomography(square, projections)
        Mr0,Mg0,Mb0= self.__CalculateShadeMatrix(image, shadeRes, points, normals)

        Mr = cv2.warpPerspective(Mr0, H, (w, h),flags=cv2.INTER_LINEAR)
        Mg = cv2.warpPerspective(Mg0, H, (w, h),flags=cv2.INTER_LINEAR)
        Mb = cv2.warpPerspective(Mb0, H, (w, h),flags=cv2.INTER_LINEAR)

        image2=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        [r,g,b]=cv2.split(image2)

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
        pass

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
