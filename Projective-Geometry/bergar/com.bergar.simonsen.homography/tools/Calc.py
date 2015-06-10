__author__ = 'bs'

import numpy as np
import cv2

def getRectangleLowerCenter(pt1, pt2):
    deltax = abs(pt2[0] - pt1[0])
    centerx = pt1[0] + deltax / 2

    return centerx, pt2[1]

def angle_cos(p0, p1, p2):
    d1, d2 = p0-p1, p2-p1
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def findSquares(img,minSize = 2000,maxAngle = 1):
    """ findSquares intend to locate rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between
    sides"""
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.08*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < maxAngle:
                squares.append(cnt)
    return squares

def DetectPlaneObject(I,minSize=1000):
    """ A simple attempt to detect rectangular
    color regions in the image"""
    HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    h = HSV[:,:,0].astype('uint8')
    s = HSV[:,:,1].astype('uint8')
    v = HSV[:,:,2].astype('uint8')

    b = I[:,:,0].astype('uint8')
    g = I[:,:,1].astype('uint8')
    r = I[:,:,2].astype('uint8')

    # use red channel for detection.
    s = (255*(r>230)).astype('uint8')
    iShow = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    cv2.imshow('ColorDetection',iShow)
    squares = findSquares(s,minSize)
    return squares