import cv2
from numpy import *

def RecordVideoFromCamera(index):
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    f,I = cap.read()
    f2,I2 = cap2.read()

    print I.shape
    H,W,Z=I.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('cameraLeft.mov', fourcc, 10.0, (W, H), True)
    writer2 = cv2.VideoWriter('cameraRight.mov', fourcc, 10.0, (W, H), True)  
    cv2.namedWindow("input")

    while(f):
        f,I = cap.read()
        f, I2 = cap2.read()
        
        if f==True:

            writer.write(I)
            writer2.write(I2)
            cv2.imshow("input", I)
            cv2.imshow("input2", I2)
            ch = cv2.waitKey(1)
            if ch == 32 or ch == 27:#  Break by SPACE key
                writer.release()
                writer2.release()
                break


RecordVideoFromCamera(1)
