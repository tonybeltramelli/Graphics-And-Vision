__author__ = 'bs'

import cv2
from config.Const import *
from tools import Utils
import numpy as np

def writeImage(I):
    fName = SAVE_FOLDER + OUTPUT_IMAGE + JPG_EXTENSION
    for i in range(MAX_FILES):
        if not Utils.doesFileExist(fName):
            cv2.imwrite(SAVE_FOLDER + fName, I)
            break
        else:
            fName = SAVE_FOLDER + OUTPUT_IMAGE + "_" + str(i) + JPG_EXTENSION

def writeHomography(H):
    fName = SAVE_FOLDER + OUTPUT_MATRIX + NPY_EXTENSION
    for i in range(MAX_FILES):
        if not Utils.doesFileExist(fName):
            np.save(fName, H)
            break;
        else:
            fName = SAVE_FOLDER + OUTPUT_MATRIX + "_" + str(i) + NPY_EXTENSION
