import argparse
import cv2
import cv
import pylab
import math
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
import numpy as np
import sys
from scipy.cluster.vq import *
from scipy.misc import imresize
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

files = [
        "Sequences/eye1.avi",
        "Sequences/eye2.avi",
        "Sequences/eye3.avi",
        "Sequences/eye4.avi",
        "Sequences/eye5.avi",
        "Sequences/eye6.avi",
        "Sequences/eye7.avi",
        "Sequences/eye8.avi",
        "Sequences/eye9.avi",
        "Sequences/eye10.avi",
        "Sequences/eye11.avi",
        "Sequences/eye12.avi",
        "Sequences/eye13.mp4",
        "Sequences/eye14.mp4",
        "Sequences/eye15.mp4",
        "Sequences/eye16.mp4",
        "Sequences/eye17.mp4",
        "Sequences/eye18.mp4",
        "Sequences/eye19.mp4",
        "Sequences/EyeBizaro.avi"
]        

inputFile = "Sequences/eye1.avi"
outputFile = "eyeTrackerResult.mp4"

#--------------------------
#         Global variable
#--------------------------
global imgOrig,leftTemplate,rightTemplate,frameNr
imgOrig = [];
#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0;

def GetPupil(gray, thr, areaMin, areaMax):
	'''Given a gray level image, gray and threshold value return a list of pupil locations'''
	#tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results
	#cv2.circle(tempResultImg,(100,200), 2, (0,0,255),4) #draw a circle
	#cv2.imshow("TempResults",tempResultImg)

	props = RegionProps()
	val,binI =cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

	cv2.imshow("Threshold",binI)
	#Calculate blobs
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	pupils = [];
	# YOUR IMPLEMENTATION HERE !!!!

        for c in contours:
                tmp = props.CalcContourProperties(c, properties=["centroid", "area", "extend"])
                x, y= tmp['Centroid']
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
	''' Given a gray level image, gray and threshold
	value return a list of glint locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	#tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

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



def update(I):
	'''Calculate the image features and display the result based on the slider values'''
	#global drawImg
	global frameNr,drawImg
	img = I.copy()
	sliderVals = getSliderVals()
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        #def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
        detectPupilKMeans(gray, K=5, distanceWeight=25, reSize=(40,40))


	# Do the magic
	pupils = GetPupil(gray,sliderVals['pupilThr'], sliderVals['minSize'], sliderVals['maxSize'])
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
		#original image
        for pupil in pupils:
                cv2.ellipse(img,pupil,(0,255,0),1)
                C = int(pupil[0][0]),int(pupil[0][1])
                cv2.circle(img,C, 2, (0,0,255),4)
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

def run(fileName,resultFile='eyeTrackingResults.avi'):

	''' MAIN Method to load the image sequence and handle user inputs'''
	global imgOrig, frameNr,drawImg
	setupWindowSliders()
	props = RegionProps();
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
#------------------------

#------------------------------------------------
#   Methods for segmentation
#------------------------------------------------
def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
	''' Detects the pupil in the image, gray, using k-means
			gray              : grays scale image
			K                 : Number of clusters
			distanceWeight    : Defines the weight of the position parameters
			reSize            : the size of the image to do k-means on
		'''
	#Resize for faster performance
	smallI = cv2.resize(gray, reSize)
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
	f = figure(1)
	imshow(labelIm)
	f.canvas.draw()
	f.show()

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
#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


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

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)

def setInputFile(n):
        global inputFile
        if n > 20:
                inputFile = files[20]
        elif n < 0:
                inputFile = files[0]
        else:
                inputFile = files[n]

#--------------------------
#         main
#--------------------------

def main():
        parser = argparse.ArgumentParser(description='Select input file')
        parser.add_argument('inputFile', metavar='N', type=int, help='A number between 1 and 20, representing the input file')
        
        args = parser.parse_args()
        setInputFile(args.inputFile)
        run(inputFile)        

if __name__ == "__main__":
        main()
