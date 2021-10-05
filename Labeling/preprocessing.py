import scipy.io
import cv2 as cv
import numpy as np

mat = scipy.io.loadmat('train0to2.mat')

labels = list(mat.values())[3:]

for i, label in enumerate(labels):
    # image = cv.imshow(label)
    threshold, thresh = cv.threshold(label, 5, 255, cv.THRESH_BINARY)
    thresh = (255-thresh) # Invert from MATLAB (have to if not image corrupted)
    cv.imwrite(f'./masked_images/image_{i}.jpg', thresh)