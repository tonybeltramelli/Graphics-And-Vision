__author__ = 'bs'

import cv2
from SIGBTools import *
import pylab
import numpy as np
import sys
from scipy.cluster.vq import *
from scipy.misc import imresize
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from Filter import *

def FilterPupilGlint(pupils,glints):
    ''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
    retval = []
    for p in pupils:
        for g in glints:
            maxLength = int(p[2]) # Max length is the radius of the pupil ellipse.
            dx = math.fabs(p[0][0] - g[0][0])
            dy = math.fabs(p[0][1] - g[0][1])
            length = math.sqrt((dx**2 + dy**2))

            if length < maxLength:
                retval.append(g)

    return retval

def GetPupil(gray, thr, areaMin, areaMax):
    props = RegionProps()
    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Threshold",binI)
    #Calculate blobs
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pupils = []

    for c in contours:
        tmp = props.CalcContourProperties(c, properties=["centroid", "area", "extend"])
        x, y = tmp['Centroid']
        area = tmp['Area']
        extend = tmp['Extend']
        if area > areaMin and area < areaMax and extend < 1:
            if len(c) >= 5:
                el = cv2.fitEllipse(c)
                pupils.append(el)
                #cv2.ellipse(tempResultImg, el, (0, 255, 0), 4)
                #cv2.circle(tempResultImg,(int(x),int(y)), 2, (0,0,255),4) #draw a circle
                #cv2.imshow("TempResults",tempResultImg)
    return pupils

def GetGlints(gray, thr, areaMin, areaMax):
    props = RegionProps()
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    val,binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

    cv2.imshow("Threshold Glint",binI)
    #cv2.imshow("Gray", gray)

    #Calculate blobs
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    glints = [];

    for c in contours:
        tmp = props.CalcContourProperties(c, properties=["centroid", "area", "extend"])
        area = tmp['Area']
        extend = tmp['Extend']
        #print tmp['Area']
        if area > areaMin and area < areaMax and extend < 1:
            if len(c) >= 5:
                el = cv2.fitEllipse(c)
                glints.append(el)
                #canny = cv2.Canny(tempResultImg, 30, 150)
                #cv2.imshow("Canny", canny)
                #x, y = tmp['Centroid']
                #cv2.circle(tempResultImg,(int(x), int(y)), 2, (0,0,255),4)
                #cv2.ellipse(tempResultImg,el,(0,255,0),1)
                #cv2.imshow("Glint detect", tempResultImg)

    return glints

def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
    ''' Detects the pupil in the image, gray, using k-means
            gray              : grays scale image
            K                 : Number of clusters
            distanceWeight    : Defines the weight of the position parameters
            reSize            : the size of the image to do k-means on
        '''
    #Resize for faster performance

    smallI = cv2.resize(gray, reSize)


    # smallI = Filter.blur(smallI)
    smallI = Filter.gaussianBlur(smallI)

    M,N = smallI.shape
    #Generate coordinates in a matrix
    X,Y = np.meshgrid(range(M),range(N))
    #Make coordinates and intensity into one vectors
    z = smallI.flatten()
    x = X.flatten()
    y = Y.flatten()
    O = len(x)
    #make a feature vectors containing (x,y,intensity)
    features = np.zeros((O,3))
    features[:,0] = z;
    features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
    features[:,2] = x/distanceWeight;
    features = np.array(features,'f')
    # cluster data
    centroids,variance = kmeans(features,K)
    #use the found clusters to map
    label,distance = vq(features,centroids)
    # re-create image from
    labelIm = np.array(np.reshape(label,(M,N)))

    tmp = centroids[[range(0, K)], [0]]
    return min(tmp[0])

    # Debugging
    # print "-----"
    # print tmp
    # print darkest
    # print "-----"

    # Show figure ?
    # f = figure(1)
    # f = figure(distanceWeight)
    # imshow(labelIm)
    # f.savefig("gaussian_" + str(K) + "_" + str(distanceWeight))
    # f.show()

def detectPupilHough(gray):
    #Using the Hough transform to detect ellipses
    blur = cv2.GaussianBlur(gray, (9,9),3)
    ##Pupil parameters
    dp = 6; minDist = 10
    highThr = 30 #High threshold for canny
    accThr = 600; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
    maxRadius = 70;
    minRadius = 20;
    #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
    circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
    #Print the circles
    gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if (circles !=None):
        #print circles
        all_circles = circles[0]
        M,N = all_circles.shape
        k=1
        for c in all_circles:
            cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
            K=k+1
        #Circle with max votes
        c=all_circles[0,:]
        cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
    cv2.imshow("hough",gColor)

def GetEyeCorners(img, leftTemplate, rightTemplate,pupilPosition=None):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    leftTemplate = cv2.cvtColor(leftTemplate, cv2.COLOR_RGB2GRAY)
    rightTemplate = cv2.cvtColor(rightTemplate, cv2.COLOR_RGB2GRAY)
    lw, lh = leftTemplate.shape[::-1]
    rw, rh = rightTemplate.shape[::-1]


    #        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)] # [green, red, blue, yellow, teal, purple]
    methods = ['cv2.TM_SQDIFF_NORMED']

    for m in range(len(methods)):
        i = img2.copy()
        method = eval(methods[m])
        color = colors[m]
        # apply template matching
        res = cv2.matchTemplate(i, leftTemplate, method)
        resRight = cv2.matchTemplate(i, rightTemplate, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        min_valr, max_valr, min_locr, max_locr = cv2.minMaxLoc(resRight)

        # if method is tm_sqdiff or tm_sqdiff_normed
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            top_leftr = min_locr
        else:
            top_left = max_loc
            top_leftr = max_locr

        bottom_right = (top_left[0] + lw, top_left[1] + lh)
        bottom_rightr = (top_leftr[0] + rw, top_leftr[1] + rh)

        cv2.rectangle(img, top_left, bottom_right, color, 1)
        cv2.rectangle(img, top_leftr, bottom_rightr, color, 1)

def GetIrisUsingThreshold(gray,pupil):
    ''' Given a gray level image, gray and threshold
    value return a list of iris locations'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass

def circularHough(gray):
    ''' Performs a circular hough transform of the image, gray and shows the  detected circles
    The circe with most votes is shown in red and the rest in green colors '''
    #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
    blur = cv2.GaussianBlur(gray, (31,31), 11)

    dp = 6; minDist = 30
    highThr = 20 #High threshold for canny
    accThr = 850; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
    maxRadius = 50;
    minRadius = 155;
    circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,maxRadius, minRadius)

    #Make a color image from gray for display purposes
    gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if (circles !=None):
        #print circles
        all_circles = circles[0]
        M,N = all_circles.shape
        k=1
        for c in all_circles:
            cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
            K=k+1
        c=all_circles[0,:]
        cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
        cv2.imshow("hough",gColor)

def GetIrisUsingNormals(gray,pupil,normalLength):
    ''' Given a gray level image, gray and the length of the normals, normalLength
     return a list of iris locations'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass

def GetIrisUsingSimplifyedHough(gray,pupil):
    ''' Given a gray level image, gray
    return a list of iris locations using a simplified Hough transformation'''
    # YOUR IMPLEMENTATION HERE !!!!
    pass
