import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread('DroneImages/test1.jpg')
image = cv.resize(image, (500,500))
image_copy = np.copy(image)

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    canny = cv.Canny(blur, 50, 150)

    return canny

canny_image = canny(image_copy)



video = cv.VideoCapture('DroneImages/video_test1.mp4')
while(video.isOpened()):
    _, frame = video.read()
    canny_image = canny(frame)
    cv.imshow('canny image', canny_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()






# cv.imshow('Sample Image', canny_image)
# cv.waitKey(0)