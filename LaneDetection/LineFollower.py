import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
hsvVals = [0, 0, 117, 179, 22, 219]

def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)

    return mask


def getContours(imgThres, img):

    # find edges of image
    contours, hierarchy = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    biggest = max(contours, key = cv.contourArea)
    x, y, w, h = cv.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2

    cv.drawContours(img, biggest, -1, (0,255,0), 7)
    cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)



while True:
    _, img = cap.read()
    img = cv.resize(img, (480,360))
    # img = cv.flip(img, 0)

    imgThres = thresholding(img)
    getContours(imgThres, img)

    cv.imshow("Output", img)
    cv.imshow("Output Thres", imgThres)
    cv.waitKey(1)