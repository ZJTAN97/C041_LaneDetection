import cv2 as cv
import numpy as np
from djitellopy import tello

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()
# drone.takeoff()


cap = cv.VideoCapture(0)
hsvVals = [0, 0, 117, 179, 22, 219]
sensors = 3
threshold = 0.2
width, height = 480, 360

sensitivity = 3 # if number is high, less sensitive

weights = [-25, -15, 0, 15, 25]
fSpeed = 15
curve = 0

def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)

    return mask


def getContours(imgThres, img):
    cx = 0
    # find edges of image
    contours, hierarchy = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours != 0):
        biggest = max(contours, key = cv.contourArea)
        x, y, w, h = cv.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2

        cv.drawContours(img, biggest, -1, (0,255,0), 7)
        cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)

    return cx


def getSensorOutput(imgThres, sensors):

    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (imgThres.shape[1] // sensors) * imgThres.shape[0]
    senOut = []
    for i, img in enumerate(imgs):
        pixelCount = cv.countNonZero(img)
        if pixelCount > threshold*totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
    #     cv.imshow(str(i), img)
    # print(senOut)
    return senOut


def sendCommands(senOut, cx):

    global curve

    ## Translation
    lr = (cx - width // 2) // sensitivity
    lr = int(np.clip(lr, -10, 10)) # clip the speed

    if lr < 2 and lr > -2: 
        lr = 0
    

    ## Rotation
    if senOut == [1, 0, 0]: curve = weights[0]
    elif senOut == [1, 1, 0]: curve = weights[1]
    elif senOut == [0, 1, 0]: curve = weights[2]
    elif senOut == [0, 1, 1]: curve = weights[3]
    elif senOut == [0, 0, 1]: curve = weights[4]

    elif senOut == [0, 0, 0]: curve = weights[2]
    elif senOut == [1, 1, 1]: curve = weights[2]
    elif senOut == [1, 0, 1]: curve = weights[2]


    drone.send_rc_control(lr, fSpeed, 0, curve)



while True:
    # _, img = cap.read()
    img = drone.get_frame_read().frame
    img = cv.resize(img, (width, height))
    img = cv.flip(img, 0)

    imgThres = thresholding(img)
    cx = getContours(imgThres, img) ## For Translation
    senOut = getSensorOutput(imgThres, sensors) ## Rotation

    sendCommands(senOut, cx)

    cv.imshow("Output", img)
    cv.imshow("Output Thres", imgThres)
    cv.waitKey(1)