__author__ = 'bs'

import cv2
from matplotlib.pyplot import *
from tools import Utils
from tools import IO
from tools import Calc
from config.Const import *
from pylab import *

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

    # Hardcoded points for simpler testing
    if PML_AUTO_POINTS:
        p1 = array([[85.74780454, 194.09873055], [258.12098948, 174.51086862], [270.36340318, 192.13994435], [89.17568038, 218.0938614]])
        p2 = array([[332.45191722, 179.62120685], [330.00343448, 100.04551778], [338.57312408, 103.71824189], [338.57312408, 179.62120685]])
        H = Utils.estimateHomography(p1, p2)

    fig = figure()
    for k in range(m):
        curData = k
        running, imgOrig = cap.read()


        # only do the calibration once
        if getPoints and not PML_AUTO_POINTS:
            H, imagePoints = Utils.getHomographyFromMouse(imgOrig, map, 4)
            getPoints = False

        if(running):
            boxes = frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]

            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
                if k == 1: # only use the "legs"
                    x1, y1 = Calc.getRectangleLowerCenter(aBox[0], aBox[1]) # calculate the center of the "legs" rectangle
                    displayTrace(map, H, x1, y1, curData, m-2)

            cv2.imshow("boxes",imgOrig)
            cv2.waitKey(DELAY)

def displayTrace(I, H, x, y, data, maxData):
    # convert x, y coordinates to homography matrix
    vec = [
        [1, 0 ,x],
        [0, 1, y],
        [0, 0, 1]
    ]

    # calculate the transformed coordinates
    p = H * vec

    # divide coordinates with the homogenous scaling factor
    x = p.item(2) / p.item(8)
    y = p.item(5) / p.item(8)

    cv2.circle(I, (int(x), int(y)), 1, (0, 0, 255), thickness=0, lineType=cv2.CV_AA, shift=0)

    # save image when sequence is done i.e., when data == maxData
    if data == maxData:
        if SAVE_MAP_IMAGE:
            IO.writeImage(I)
        if SAVE_H_M_G:
            IO.writeHomography(H)

    # TODO: Write video

    cv2.imshow("map", I)

def frameTrackingData2BoxData(data):

    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)
    return boxes