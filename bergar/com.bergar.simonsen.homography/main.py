__author__ = 'bs'

import cv2
from matplotlib import *
from tools import IO
import numpy as np
from matplotlib.pyplot import *
from config.Const import *
from tools import Utils
from tools import Calc
import personMapLocation as pml
import textureMapping as tm


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
                 #TODO Do texturemap here!!!!

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



def realisticTexturemap(scale,point,map):
    #H = np.load('H_G_M')
    print "Not implemented yet\n"*30








def main():
    # pml.showFloorTrackingData()
    # tm.simpleTextureMap()
    # tm.textureMapGroundFloor()
    # realisticTexturemap(0,0,0)
    tm.texturemapGridSequence()

if __name__ == "__main__":
    main()