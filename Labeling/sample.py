import cv2 as cv
import numpy as np

# For rough work purposes

img = cv.imread('masked_images/image_0.jpg')
img = cv.resize(img, (512,512))

cv.imshow('image', img)

ret, thresh1 = cv.threshold(img, 10, 255, cv.THRESH_BINARY)
gray = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)

kernel = np.ones((3,3), np.uint8)
erode = cv.dilate(gray, kernel, iterations=1)
cv.imshow('gray scaled', erode)

cv.waitKey(0)