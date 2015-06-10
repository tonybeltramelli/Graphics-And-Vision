__author__ = 'bs'

import cv2
import numpy as np
from config.Const import *
from tools import Utils
from matplotlib.pyplot import figure

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

def textureMapGroundFloor():
    #Load videodata
    # logo = cv2.imread(ITU_LOGO)
    texture = cv2.imread(TEXTURE)
    fn = GROUND_FLOOR_VIDEO

    cap = cv2.VideoCapture(fn)

    #load Tracking data
    running, imgOrig = cap.read()

    H,Points  = Utils.getHomographyFromMouse(texture, imgOrig, -1)
    h, w,d = imgOrig.shape

    while(cap.isOpened()):
        ret, frame = cap.read()

        try:
            overlay = cv2.warpPerspective(texture, H,(w, h))
            wFirst = 0.9
            wSecond = 0.1
            gamma = 9
            M = cv2.addWeighted(frame, wFirst, overlay, wSecond, gamma)
        except:
            break

        cv2.imshow("Overlayed Image",M)

        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
            cv2.waitKey(DELAY)