__author__ = 'bs'

import cv2
from matplotlib import *
from matplotlib.pyplot import *
from config.Const import *
from tools import Utils
from tools import Calc
import numpy as np

def texturemapObjectSequence():
    """ Poor implementation of simple texturemap """

    fn = BOOK_3
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread(ITU_LOGO)
    #texture = cv2.transpose(texture)
    mTex,nTex,t = texture.shape

    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    print running
    while(running):
        for t in range(20):
            running, imgOrig = cap.read()

        if(running):
            squares = Calc.DetectPlaneObject(imgOrig)

            for sqr in squares:
                 #Do texturemap here!!!!
                 #TODO

                if(drawContours):
                    for p in sqr:
                        cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0))


            if(drawContours and len(squares)>0):
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)

def showImageandPlot(N):
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    # I = cv2.imread('groundfloor.bmp')
    I = cv2.imread(ITU_MAP)
    drawI = I.copy()
    #make figure and two subplots
    fig = figure(1)
    ax1  = subplot(1,2,1)
    ax2  = subplot(1,2,2)
    ax1.imshow(I)
    ax2.imshow(drawI)
    ax1.axis('image')
    ax1.axis('off')
    points = fig.ginput(5)
    fig.hold('on')

    for p in points:
        #Draw on figure
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        #Draw in image
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),10)
    # ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered
    show()
    savefig('somefig.jpg')
    cv2.imwrite("drawImage.jpg", drawI)

def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = GRID_1
    cap = cv2.VideoCapture(fn)
    drawContours = True

    texture = cv2.imread(ITU_LOGO)
    texture = cv2.pyrDown(texture)


    mTex,nTex,t = texture.shape

    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    cv2.imshow("win2",imgOrig)

    pattern_size = (9, 6)

    idx = [0,8,45,53]
    while(running):
    #load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)

                for t in idx:
                    cv2.circle(imgOrig,(int(corners[t,0,0]),int(corners[t,0,1])),10,(255,t,t))
            cv2.imshow("win2",imgOrig)
            cv2.waitKey(1)

def realisticTexturemap(scale,point,map):
    #H = np.load('H_G_M')
    print "Not implemented yet\n"*30


def simpleTextureMap():

    I1 = cv2.imread(ITU_LOGO)
    I2 = cv2.imread(ITU_MAP)

    #Print Help
    H,Points  = Utils.getHomographyFromMouse(I1,I2,4)
    h, w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5,0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)


def frameTrackingData2BoxData(data):

    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)
    return boxes

def displayTrace(H, x, y):

    # A = np.array([[86.23750109, 196.05751674], [77.42296322, 97.62851056], [ 141.57321103, 88.8139727 ], [ 165.07864534, 184.30479958]])
    # B = np.array([[ 330.00343448, 179.62120685], [239.40957308, 182.06968959], [238.18533171, 139.22124163], [ 330.00343448, 141.66972437]])

    # print H
    # vec = [x, y, 1]
    vec = [
        [1, 0 ,x],
        [0, 1, y],
        [0, 0, 1]
    ]
    p = H * vec

    # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    # print p
    # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #
    # print str(p.item(2)) + " " + str(p.item(5))
    # print str(p.item(2)/p.item(8)) + " " + str(p.item(5)/p.item(8))

    x = p.item(2) / p.item(8)
    y = p.item(5) / p.item(8)


    # print "---"
    # print type(p)
    # print "---"
    # print p[0][1]
    # return p[0][2], p[1][2]
    # return x + p[0][2], y + p[1][2]
    # return p.item(2), p.item(5)
    return x, y

def test():
    map = cv2.imread(ITU_MAP)
    map_rotate = cv2.imread("../Images/ITUMapCopy.bmp")

    H, imagePoints = Utils.getHomographyFromMouse(map, map_rotate, 4)

    for p in imagePoints:
        colr = (255, 0, 0)
        for x in p:
            x2, y2 = displayTrace(H, x[0], x[1])
            cv2.circle(map, (int(x2), int(y2)), 3, colr, thickness=1, lineType=8, shift=0)
            cv2.circle(map_rotate, (int(x2), int(y2)), 3, colr, thickness=1, lineType=8, shift=0)
        colr = (0, 255, 0)

    # x2, y2 = displayTrace(H, x1, y1)
    # print "new coords: (" + str(x2) + ", " + str(y2) + ")"
    # cv2.circle(map, (int(x2), int(y2)), 1, (0, 0, 255), thickness=1, lineType=8, shift=0)

    cv2.imshow("map", map)
    cv2.imshow("map rotate", map_rotate)

    cv2.waitKey(0)

def showFloorTrackingData():
    #Load videodata
    map = cv2.imread(ITU_MAP)
    fn = GROUND_FLOOR_VIDEO
    cap = cv2.VideoCapture(fn)

    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt(TRACKING_DATA)
    m,n = dataFile.shape

    getPoints = True

    H = []
    imagePoints = []

    fig = figure()
    for k in range(m):
        running, imgOrig = cap.read()
        if getPoints:
            H, imagePoints = Utils.getHomographyFromMouse(imgOrig, map, 4)
            getPoints = False
            # print H
            # print imagePoints
            # print type(H)
            # print "-------------------------"
            # print H
            # print "-------------------------"
            # print imagePoints
            # print "-------------------------"

        if(running):
            boxes = frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]

            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
                if k == 1: # only use the legs
                    x1, y1 = Calc.getRectangleLowerCenter(aBox[0], aBox[1])
                    # x1 = aBox[1][0]
                    # y1 = aBox[1][1]
                    colr = (255, 0, 0)
                    # for p in imagePoints:
                    #     for x in p:
                    #         print ">>>>>" + str(x[0]) + ", " + str(x[1])
                    #         x2, y2 = displayTrace(H, x[0], x[1])
                    #         print "-----------" + str(x2) + ", " + str(y2)
                    #         cv2.circle(map, (int(x2), int(y2)), 3, colr, thickness=1, lineType=8, shift=0)
                    #     colr = (0, 255, 0)
                    x2, y2 = displayTrace(H, x1, y1)
                    # print "new coords: (" + str(x2) + ", " + str(y2) + ")"
                    cv2.circle(map, (int(x2), int(y2)), 1, (0, 0, 255), thickness=0, lineType=8, shift=0)



            cv2.imshow("boxes",imgOrig)
            cv2.imshow("map", map)
            cv2.waitKey(20)
            # cv2.waitKey(0)

def main():
    showFloorTrackingData()
    # simpleTextureMap()
    # realisticTexturemap(0,0,0)
    # texturemapGridSequence()

    # test()

if __name__ == "__main__":
    main()