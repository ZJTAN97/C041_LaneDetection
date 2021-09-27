from djitellopy import tello
import cv2 as cv
import numpy as np

frameWidth = 480
frameHeight = 360

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

def empty(a):

    pass

cv.namedWindow("HSV")
cv.resizeWindow("HSV", 640, 240)
cv.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

cap = cv.VideoCapture(0)

frameCounter = 0

while True:

    img = me.get_frame_read().frame

    # _, img = cap.read()

    # img = cv.imread('test2.jpg')
    # img = cv.resize(img, (frameWidth, frameHeight))

    img = cv.flip(img,0)

    imgHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("HUE Min", "HSV")
    h_max = cv.getTrackbarPos("HUE Max", "HSV")
    s_min = cv.getTrackbarPos("SAT Min", "HSV")
    s_max = cv.getTrackbarPos("SAT Max", "HSV")
    v_min = cv.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv.getTrackbarPos("VALUE Max", "HSV")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHsv, lower, upper)
    result = cv.bitwise_and(img, img, mask=mask)

    print(f'[{h_min},{s_min},{v_min},{h_max},{s_max},{v_max}]')

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv.imshow('Horizontal Stacking', hStack)

    if cv.waitKey(1) and 0xFF == ord('q'):

        break

cap.release()
cv.destroyAllWindows()