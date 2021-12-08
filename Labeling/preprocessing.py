import scipy.io
import cv2 as cv
import numpy as np

mat = scipy.io.loadmat('1to11.mat')

labels = list(mat.values())[3:]

for i, img in enumerate(labels):
    # image = cv.imshow(label)
    threshold, thresh = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    dilate = cv.dilate(gray, kernel, iterations=2)
    cv.imwrite(f'./masked_images/image_{i}.jpg', dilate)