__author__ = 'bs'

from SIGBTools import *
from Tracker import *
import numpy as np
import cv as cv

# -----------------------------
# Global variables
# -----------------------------

files = [
    "eye1.avi",
    "eye2.avi",
    "eye3.avi",
    "eye4.avi",
    "eye5.avi",
    "eye6.avi",
    "eye7.avi",
    "eye8.avi",
    "eye9.avi",
    "eye10.avi",
    "eye11.avi",
    "eye12.avi",
    "eye13.mp4",
    "eye14.mp4",
    "eye15.mp4",
    "eye16.mp4",
    "eye17.mp4",
    "eye18.mp4",
    "eye19.mp4",
    "EyeBizaro.avi"
]

dir = "../Sequences/"
inputFile = dir + files[10]
outputFile = dir + "eyeTrackerResult.mp4"

imgOrig = []

#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0

def setText(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

def update(I):
    global frameNr,drawImg
    img = I.copy()
    sliderVals = getSliderVals()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    kmeanTresh = detectPupilKMeans(gray, K=12, distanceWeight=2, reSize=(40,40))

    # Do the magic
    pupils = GetPupil(gray,sliderVals['pupilThr'], sliderVals['minSize'], sliderVals['maxSize'])
    pupils2 = GetPupil(gray,kmeanTresh, sliderVals['minSize'], sliderVals['maxSize'])
    glints = GetGlints(gray,sliderVals['glintThr'], 0, 150)
    glints = FilterPupilGlint(pupils,glints)

    #Do template matching
    global leftTemplate
    global rightTemplate
    if len(leftTemplate) > 0 and len(rightTemplate) > 0:
        GetEyeCorners(img, leftTemplate, rightTemplate)

    #Display results
    global frameNr,drawImg
    x,y = 10,10
    #setText(img,(x,y),"Frame:%d" %frameNr)
    sliderVals = getSliderVals()

    # for non-windows machines we print the values of the threshold in the original image
    if sys.platform != 'win32':
        step=18
        cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
        cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
    cv2.imshow('Result',img)

    #Uncomment these lines as your methods start to work to display the result in the
    #original image using blob detection and sliders for treshold
    for pupil in pupils:
        cv2.ellipse(img,pupil,(0,255,0),1)
        C = int(pupil[0][0]),int(pupil[0][1])
        cv2.circle(img,C, 2, (0,0,255),4)
        for glint in glints:
            C = int(glint[0][0]),int(glint[0][1])
            cv2.circle(img,C, 2,(255,0,255),5)
    cv2.imshow("Result", img)

    # draw results using kmeans clustering
    for pupil in pupils2:
        cv2.ellipse(img,pupil,(255,255,0),1)
        C = int(pupil[0][0]),int(pupil[0][1])
        cv2.circle(img,C, 2, (0,255,255),4)
        for glint in glints:
            C = int(glint[0][0]),int(glint[0][1])
            cv2.circle(img,C, 2,(255,0,255),5)
    cv2.imshow("Result", img)



    #For Iris detection - Week 2
    #circularHough(gray)

    #copy the image so that the result image (img) can be saved in the movie
    drawImg = img.copy()



def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"
    print "r: reload video"
    print 'm: Mark region when the video has paused'
    print 's: toggle video  writing'
    print 'c: close video sequence'

def onSlidersChange(dummy=None):
    ''' Handle updates when slides have changed.
     This  function only updates the display when the video is put on pause'''
    global imgOrig;
    sv = getSliderVals()
    if(not sv['Running']): # if pause
        update(imgOrig)

def setupWindowSliders():
    ''' Define windows for displaying the results and create trackbars'''
    cv2.namedWindow("Result")
    cv2.namedWindow('Threshold')
    #cv2.namedWindow("TempResults")
    #Threshold value for the pupil intensity
    cv2.createTrackbar('pupilThr','Threshold', 90, 255, onSlidersChange)
    #Threshold value for the glint intensities
    cv2.createTrackbar('glintThr','Threshold', 240, 255,onSlidersChange)
    #define the minimum and maximum areas of the pupil
    cv2.createTrackbar('minSize','Threshold', 20, 200, onSlidersChange)
    cv2.createTrackbar('maxSize','Threshold', 200,200, onSlidersChange)
    #Value to indicate whether to run or pause the video
    cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)

def getSliderVals():
    '''Extract the values of the sliders and return these in a dictionary'''
    sliderVals={}
    sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Threshold')
    sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Threshold')
    sliderVals['minSize'] = 50*cv2.getTrackbarPos('minSize', 'Threshold')
    sliderVals['maxSize'] = 50*cv2.getTrackbarPos('maxSize', 'Threshold')
    sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')
    return sliderVals

def run(fileName,resultFile='eyeTrackingResults.avi'):

    ''' MAIN Method to load the image sequence and handle user inputs'''
    global imgOrig, frameNr,drawImg
    setupWindowSliders()
    props = RegionProps()
    cap,imgOrig,sequenceOK = getImageSequence(fileName)
    videoWriter = 0

    frameNr =0
    if(sequenceOK):
        #detectPupilKMeans(gray = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY))
        update(imgOrig)
    printUsage()
    frameNr=0;
    saveFrames = False

    while(sequenceOK):
        sliderVals = getSliderVals();
        frameNr=frameNr+1
        ch = cv2.waitKey(1)
        #Select regions
        if(ch==ord('m')):
            if(not sliderVals['Running']):
                roiSelect=ROISelector(imgOrig)
                pts,regionSelected= roiSelect.SelectArea('Select left eye corner',(400,200))

                roiSelect2=ROISelector(imgOrig)
                pts2,regionSelected2= roiSelect2.SelectArea('Select right eye corner',(400,200))

                # TODO: Optimize
                global leftTemplate
                global rightTemplate
                if(regionSelected):
                    leftTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
                if(regionSelected2):
                    rightTemplate = imgOrig[pts2[0][1]:pts2[1][1],pts2[0][0]:pts2[1][0]]

        if ch == 27:
            break
        if (ch==ord('s')):
            if((saveFrames)):
                videoWriter.release()
                saveFrames=False
                print "End recording"
            else:
                imSize = np.shape(imgOrig)
                videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
                saveFrames = True
                print "Recording..."



        if(ch==ord('q')):
            break
        if(ch==32): #Spacebar
            sliderVals = getSliderVals()
            cv2.setTrackbarPos('Stop/Start','Threshold',not sliderVals['Running'])
        if(ch==ord('r')):
            frameNr =0
            sequenceOK=False
            cap,imgOrig,sequenceOK = getImageSequence(fileName)
            (imgOrig)
            sequenceOK=True

        sliderVals=getSliderVals()
        if(sliderVals['Running']):
            sequenceOK, imgOrig = cap.read()
            if(sequenceOK): #if there is an image
                #detectPupilKMeans(gray = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2GRAY))
                update(imgOrig)
            if(saveFrames):
                videoWriter.write(drawImg)
    if(videoWriter!=0):
        videoWriter.release()
    print "Closing videofile..."

def main():
    run(inputFile)

if __name__ == "__main__":
    main()