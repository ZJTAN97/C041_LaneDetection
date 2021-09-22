import scipy.io
import cv2 as cv

mat = scipy.io.loadmat('label_workspace.mat')

labels = list(mat.values())[3:]

for i, label in enumerate(labels):
    # image = cv.imshow(label)
    threshold, thresh = cv.threshold(label, 20, 255, cv.THRESH_BINARY)
    cv.imwrite(f'test_{i}.jpg', thresh)

# label = mat['maskedImage3']
# gray = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
# threshold, thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)

# # cv.imshow('test', thresh)
# # cv.waitKey(0)

# cv.imwrite('test2.jpg', thresh)