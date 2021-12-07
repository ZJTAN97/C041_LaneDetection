import cv2 as cv

img = cv.imread('picture1.png')
# img = cv.resize(img, (512,512))
canny = cv.Canny(img, 125, 175)

cv.imshow('wdf', img)
cv.imshow('image', canny)
cv.waitKey(0)