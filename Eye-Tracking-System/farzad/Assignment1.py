import cv2
#import cv
import pylab
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
import numpy as np
import sys
from scipy.cluster.vq import *
from scipy.misc import imresize
from matplotlib.pyplot import *
import scipy




inputFile = "Sequences/eye1.m4v"
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
img = []



def GetPupil(gray,thr,minS,maxS):
	'''Given a gray level image, gray and threshold value return a list of pupil locations'''
	tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results
	#cv2.circle(tempResultImg,(100,200), 2, (0,0,255),4) #draw a circle
	#cv2.imshow("TempResults",tempResultImg)

	props = RegionProps()
	val,binI =cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow("Threshold",binI)
	#Calculate blobs
	_, contours, hierarchy= cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	pupils = [];
	pupilEllipses = [];
	Centroids = [];
	# YOUR IMPLEMENTATION HERE !!!!
	
	#CalcContourProperties
	
	for cnt in contours:
		vals = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull'])
		if vals['Area']> minS and vals['Area']< maxS:
			if vals['Extend'] < 1:
				
				pupils.append(cnt)
				#print vals['Centroid']
				#cv2.circle(tempResultImg,(int(vals['Centroid'][0]),int(vals['Centroid'][1])), 2, (255,0,0),-1) #draw a circle
				Centroids.append(vals['Centroid'])
				pupilEllipse = cv2.fitEllipse(cnt)
				pupilEllipses.append(pupilEllipse)
				#cv2.ellipse(tempResultImg,ellipse,(0,255,0),2)
	#cv2.imshow("TempResults",tempResultImg)
	return (pupils,pupilEllipses,Centroids)

def GetGlints(gray,thr,minS,maxS):
	''' Given a gray level image, gray and threshold
	value return a list of glint locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results
	#cv2.circle(tempResultImg,(100,200), 2, (0,0,255),4) #draw a circle
	#cv2.imshow("TempResults",tempResultImg)

	props = RegionProps()
	val,binI =cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
	cv2.imshow("ThresholdInverse",binI)
	#Calculate blobs
	_, contours, hierarchy= cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	glints = [];
	glintEllipses = [];
	Centroids = [];
	# YOUR IMPLEMENTATION HERE !!!!

	#CalcContourProperties

	for cnt in contours:
		vals = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull'])
		if vals['Area']> minS and vals['Area']< maxS:
			if vals['Extend'] < 1:

				glints.append(cnt)
				#print vals['Centroid']
				#cv2.circle(tempResultImg,(int(vals['Centroid'][0]),int(vals['Centroid'][1])), 2, (255,0,0),-1) #draw a circle
				Centroids.append(vals['Centroid'])
				glintEllipse = cv2.fitEllipse(cnt)
				glintEllipses.append(glintEllipse)
				#cv2.ellipse(tempResultImg,ellipse,(0,255,0),2)
	#cv2.imshow("TempResultsglint",tempResultImg)
	return (glints,glintEllipses,Centroids)

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

def GetEyeCorners(leftTemplate, rightTemplate,pupilPosition=None):
	pass

def FilterPupilGlint(pupilcenters,glintcenters,minDisG,maxDisG):
	''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
	for pupilcenter1 in pupilcenters:
		for pupilcenter2 in pupilcenters:
			if minDisG< Distance(pupilcenter1, pupilcenter2) < maxDisG:
				return (pupilcenter1, pupilcenter2)

def Distance(u,v):
	#dst= scipy.distance.euclidean(u,v)
	distance=np.linalg.norm(np.array(u)-np.array(v))
	return distance

def update(I):
	'''Calculate the image features and display the result based on the slider values'''
	#global drawImg
	global frameNr,drawImg
	img = I.copy()
	sliderVals = getSliderVals()
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# Do the magic
	#(pupils,pupilEllipses,Centroids) = GetPupil(gray,sliderVals['pupilThr'],sliderVals['minSize'],sliderVals['maxSize'])
	#(glints,glintEllipses,glintCentroids) = GetGlints(gray,sliderVals['glintThr'],sliderVals['minSizeG'],sliderVals['maxSizeG'])
	#FilterPupilGlint(Centroids,glintCentroids,1,100000000)



	#Kmeans implementation
	#detectPupilKMeans(gray,20,2,reSize=(40,40))
	
	
	#Hough Transform for pupils
	#detectPupilHough(gray,sliderVals['minDist'],sliderVals['highThr'],sliderVals['accThr'],sliderVals['minRadius'],sliderVals['maxRadius'])

	#Hough Transform for Iris
	detectIrisHough(gray,sliderVals['minDistI'],sliderVals['highThrI'],sliderVals['accThrI'],sliderVals['minRadiusI'],sliderVals['maxRadiusI'])

	#Do template matching
	global leftTemplate
	global rightTemplate
	GetEyeCorners(leftTemplate, rightTemplate)
	#Display results
	global frameNr,drawImg
	x,y = 10,10
	#setText(img,(x,y),"Frame:%d" %frameNr)
	sliderVals = getSliderVals()

	# for non-windows machines we print the values of the threshold in the original image
	if sys.platform != 'win32':
		step=18
		cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
		cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
	#cv2.imshow('Result',img)

		#Uncomment these lines as your methods start to work to display the result in the
		#original image
		# for pupil in pupils:
	"""if pupilEllipses is not None and Centroids is not None:
		
		for ellipse in pupilEllipses:
			
			cv2.ellipse(img,ellipse,(0,255,0),1)
		for centroid in Centroids:
			
			C = (int(centroid[0]),int(centroid[1]))
			cv2.circle(img,C, 2, (0,0,255),4)
	for glint in glintCentroids:
		C = int(glint[0]),int(glint[1])
		cv2.circle(img,C, 2,(255,0,255),5)
		#     cv2.imshow("Result", img)

		#For Iris detection - Week 2
		#circularHough(gray)
		
	"""
	cv2.imshow('Result',img)
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
				ptsL,regionSelectedL= roiSelect.SelectArea('Select left eye corner',(400,200))
				if(regionSelectedL):
					leftTemplate = imgOrig[ptsL[0][1]:ptsL[1][1],ptsL[0][0]:ptsL[1][0]]
				
				ptsR,regionSelectedR= roiSelect.SelectArea('Select right eye corner',(600,300))
				if(regionSelectedR):
					rightTemplate = imgOrig[ptsR[0][1]:ptsR[1][1],ptsR[0][0]:ptsR[1][0]]


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
			update(imgOrig)
			sequenceOK=True

		sliderVals=getSliderVals()
		if(sliderVals['Running']):
			sequenceOK, imgOrig = cap.read()
			if(sequenceOK): #if there is an image
				update(imgOrig)
			if(saveFrames):
				videoWriter.write(drawImg)
	if(videoWriter!=0):
		videoWriter.release()
        print "Closing videofile..."
#------------------------

""""def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
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
"""
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

def detectPupilHough(gray,minDist,highThr,accThr,minRadius,maxRadius):
	#Using the Hough transform to detect ellipses
	blur = cv2.GaussianBlur(gray, (9,9),3)
	##Pupil parameters
	dp = 6
	#minDist = 10
	#highThr = 10 #High threshold for canny
	#accThr = 200; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	#maxRadius = 50;
	#minRadius = 20;
	#See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
	circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
	#Print the circles
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
		#print circles
		all_circles = circles[0]
		M,N = all_circles.shape
		k=1
		for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			#setText(gColor, (int(c[0]),int(c[1])), "Radius:" + str(c[2]))
			cv2.putText(gColor,"Radius:" + str(c[2]), (int(c[0]),int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
			K=k+1
			#Circle with max votes
		c=all_circles[0,:]
		cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
	#text labels
	
	setText(gColor, (10, 10), "minDist:" + str(minDist) + " highThr:" + str(highThr) +" accThr:" + str(accThr) +" minRadius:" + str(minRadius) +" maxRadius:" + str(maxRadius) )
	
	cv2.imshow("hough",gColor)
	
def detectIrisHough(gray,minDist,highThr,accThr,minRadius,maxRadius):
	#Using the Hough transform to detect ellipses
	blur = cv2.GaussianBlur(gray, (9,9),3)
	##Pupil parameters
	dp = 6
	#minDist = 10
	#highThr = 10 #High threshold for canny
	#accThr = 200; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	#maxRadius = 50;
	#minRadius = 20;
	#See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
	circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
	#Print the circles
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
		#print circles
		all_circles = circles[0]
		M,N = all_circles.shape
		k=1
		for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			#setText(gColor, (int(c[0]),int(c[1])), "Radius:" + str(c[2]))
			cv2.putText(gColor,"Radius:" + str(c[2]), (int(c[0]),int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
			K=k+1
			#Circle with max votes
		c=all_circles[0,:]
		cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
	#text labels

	setText(gColor, (10, 10), "minDist:" + str(minDist) + " highThr:" + str(highThr) +" accThr:" + str(accThr) +" minRadius:" + str(minRadius) +" maxRadius:" + str(maxRadius) )

	cv2.imshow("houghI",gColor)
#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))


def setupWindowSliders():
	''' Define windows for displaying the results and create trackbars'''
	cv2.namedWindow("Result")
	cv2.namedWindow('Threshold')
	cv2.namedWindow("TempResults")
	cv2.namedWindow("hough")
	cv2.namedWindow("houghI")
	#Threshold value for the pupil intensity
	cv2.createTrackbar('pupilThr','Threshold', 90, 255, onSlidersChange)
	#Threshold value for the glint intensities
	cv2.createTrackbar('glintThr','Threshold', 240, 255,onSlidersChange)
	#define the minimum and maximum areas of the pupil
	cv2.createTrackbar('minSize','Threshold', 20, 200, onSlidersChange)
	cv2.createTrackbar('maxSize','Threshold', 200,200, onSlidersChange)
	cv2.createTrackbar('minSizeG','Threshold', 20, 200, onSlidersChange)
	cv2.createTrackbar('maxSizeG','Threshold', 30,200, onSlidersChange)	
	#Value to indicate whether to run or pause the video
	cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)
	
	#Hough Transform pupils
	
	cv2.createTrackbar('minDist','hough', 1,100, onSlidersChange)
	cv2.createTrackbar('highThr','hough', 1,50, onSlidersChange)
	cv2.createTrackbar('accThr','hough', 50,800, onSlidersChange)
	cv2.createTrackbar('minRadius','hough', 10,100, onSlidersChange)
	cv2.createTrackbar('maxRadius','hough', 20,300, onSlidersChange)
	
	
	#Hough Transform Iris
	
	cv2.createTrackbar('minDistI','houghI', 10,200, onSlidersChange)
	cv2.createTrackbar('highThrI','houghI', 10,100, onSlidersChange)
	cv2.createTrackbar('accThrI','houghI', 50,800, onSlidersChange)
	cv2.createTrackbar('minRadiusI','houghI', 100,180, onSlidersChange)
	cv2.createTrackbar('maxRadiusI','houghI', 110,180, onSlidersChange)
	
	

def getSliderVals():
	'''Extract the values of the sliders and return these in a dictionary'''
	sliderVals={}
	sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Threshold')
	sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Threshold')
	sliderVals['minSize'] = 50*cv2.getTrackbarPos('minSize', 'Threshold')
	sliderVals['maxSize'] = 50*cv2.getTrackbarPos('maxSize', 'Threshold')
	sliderVals['minSizeG'] = cv2.getTrackbarPos('minSizeG', 'Threshold')
	sliderVals['maxSizeG'] = 50*cv2.getTrackbarPos('maxSizeG', 'Threshold')
	sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')
	
	#Hough pupils
	sliderVals['minDist'] = cv2.getTrackbarPos('minDist', 'hough')
	sliderVals['highThr'] = cv2.getTrackbarPos('highThr', 'hough')
	sliderVals['accThr'] = cv2.getTrackbarPos('accThr', 'hough')
	sliderVals['minRadius'] = cv2.getTrackbarPos('minRadius', 'hough')
	sliderVals['maxRadius'] = cv2.getTrackbarPos('maxRadius', 'hough')
	
	
	#Hough Iriss
	sliderVals['minDistI'] = cv2.getTrackbarPos('minDistI', 'houghI')
	sliderVals['highThrI'] = cv2.getTrackbarPos('highThrI', 'houghI')
	sliderVals['accThrI'] = cv2.getTrackbarPos('accThrI', 'houghI')
	sliderVals['minRadiusI'] = cv2.getTrackbarPos('minRadiusI', 'houghI')
	sliderVals['maxRadiusI'] = cv2.getTrackbarPos('maxRadiusI', 'houghI')

	
	
	
	
	
	
	return sliderVals

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)

#--------------------------
#         main
#--------------------------
run(inputFile)