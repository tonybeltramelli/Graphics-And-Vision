import cv2
import numpy as np

class RegionProps:
    def __calcArea(self,m,c):
        return cv2.contourArea(c) #,m['m00']

    def __calcLength(self,c):
        return cv2.arcLength(c, True)

    def __calcPerimiter(self,c):
         return cv2.arcLength(c,True)

    def __calcBoundingBox(self,c):
        return cv2.boundingRect(c)

    def __calcCentroid(self,m):
        if(m['m00']!=0):
            retVal =  ( m['m10']/m['m00'],m['m01']/m['m00'] )
        else:
            retVal = (-1,-1)
        return retVal

    def __calcEquivDiameter(self,m):
        Area = self.__calcArea(m)
        return np.sqrt(4*Area/np.pi)

    def __calcExtend(self,m,c):
        Area = self.__calcArea(m,c)
        BoundingBox = self.__calcBoundingBox(c)
        return Area/(BoundingBox[2]*BoundingBox[3])

    def __calcConvexHull(self,m,c):
         #try:
             CH = cv2.convexHull(c)
             #ConvexArea  = cv2.contourArea(CH)
             #Area =  self.__calcArea(m,c)
             #Solidity = Area/ConvexArea
             return {'ConvexHull':CH} #{'ConvexHull':CH,'ConvexArea':ConvexArea,'Solidity':Solidity}
         #except:
         #    print "stuff:", type(m), type(c)

    def CalcContourProperties(self,contour,properties=[]):
        failInInput = False;
        propertyList=[]
        contourProps={};
        for prop in properties:
            prop = str(prop).lower()
            m = cv2.moments(contour) #Always call moments
            if (prop=='area'):
                contourProps.update({'Area':self.__calcArea(m,contour)});
            elif (prop=="boundingbox"):
                contourProps.update({'BoundingBox':self.__calcBoundingBox(contour)});
            elif (prop=="length"):
                contourProps.update({'Length':self.__calcLength(contour)});
            elif (prop=="centroid"):
                contourProps.update({'Centroid':self.__calcCentroid(m)});
            elif (prop=="moments"):
                contourProps.update({'Moments':m});
            elif (prop=="perimiter"):
                contourProps.update({'Perimiter':self.__calcPerimiter(contour)});
            elif (prop=="equivdiameter"):
                contourProps.update({'EquivDiameter':self.__calcEquiDiameter(m,contour)});
            elif (prop=="extend"):
                contourProps.update({'Extend':self.__calcExtend(m,contour)});
            elif (prop=="convexhull"): #Returns the dictionary
                contourProps.update(self.__calcConvexHull(m,contour));
            elif (prop=="isConvex"):
                    contourProps.update({'IsConvex': cv2.isContourConvex(contour)});
            elif failInInput:
                    pass
            else:
                print "--"*20
                print "*** PROPERTY ERROR "+ prop+" DOES NOT EXIST ***"
                print "THIS ERROR MESSAGE WILL ONLY BE PRINTED ONCE"
                print "--"*20
                failInInput = True
        return contourProps