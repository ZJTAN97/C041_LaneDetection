import cv2 as cv
import numpy as np


hsvVals = [0, 0, 117, 179, 22, 219]


def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)

    return mask


img = cv.imread('./data/sample_train/image_0.jpg')
img = cv.resize(img, (512,512))

mask = thresholding(img)

cv.imshow('test', mask)
cv.waitKey(0)