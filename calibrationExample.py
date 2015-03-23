import numpy as np
import SIGBTools      
import cv2
from SIGBTools import Camera
def calibrationExample():
    camNum =0           # The number of the camera to calibrate
    nPoints = 5         # number of images used for the calibration (space presses)
    patternSize=(9,6)   #size of the calibration pattern
    saveImage = False

    calibrated, camera_matrix,dist_coefs,rms = SIGBTools.calibrateCamera(camNum,nPoints,patternSize,saveImage)
    K = camera_matrix
    cam1 =Camera( np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )) )
    cam1.factor()
    #Factor projection matrix into intrinsic and extrinsic parameters
    print "K=", cam1.K
    print "R=", cam1.R
    print "t", cam1.t
    
    if (calibrated):
        capture = cv2.VideoCapture(camNum)
        running = True
        while running:
            running, img =capture.read()
            imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ch = cv2.waitKey(1)
            if(ch==27) or (ch==ord('q')): #ESC
                running = False
            img=cv2.undistort(img, camera_matrix, dist_coefs )
            found,corners=cv2.findChessboardCorners(imgGray, patternSize  )
            if (found!=0):
                cv2.drawChessboardCorners(img, patternSize, corners,found)
            cv2.imshow("Calibrated",img)
                
calibrationExample()