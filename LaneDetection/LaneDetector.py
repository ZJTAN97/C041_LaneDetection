from djitellopy import tello
import cv2 as cv
import numpy as np

FRAME_WIDTH = 480
FRAME_HEIGHT = 360
THRESHOLD = 0.2

SENSITIVITY = 3
WEIGHTS = [-25, -15, 0, 15, 15]
FORWARD_SPEED = 15
CURVE = 0


# drone = tello.Tello()

hsvVals = [0, 0, 117, 179, 22, 219]

def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)

    return mask


def drone_connect(drone):
    """
    To connect to Tello Drone via djitellopy API
    """
    drone.connect()
    print('--- Drone Connected ---')
    print(f'-- Drone Battery {drone.get_battery()}% ---')


def get_translation(processedImg, img):
    """
    To get the contours from the prediction of a trained model
    Contours will handle the translation motion of the drone
    """
    translation = 0

    ## Put predictions here
    contours, hierarchy = cv.findContours(processedImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(biggest)
        translation_x = x + w // 2
        translation_y = y + h // 2

        cv.drawContours(img, biggest, -1, (0, 255, 0), 7)
        cv.circle(img, (translation_x, translation_y), 10, (255, 0, 0), cv.FILLED)
        cv.imshow('test',img)
    
    return translation

    
def get_rotation(processedImg, sensors):
    """
    Args:
    processedImg --> Preprocesed image
    sensors --> how sensitive the drone will be in terms of rotation
    
    Returns:
    rotation_vector of the drone
    """
    img_split = np.hsplit(processedImg, sensors)
    total_pixels = (processedImg.shape[1] // sensors) * processedImg.shape[0]
    rotation_vector = []

    for img in img_split:
        pixel_count = cv.countNonZero(img)
        if pixel_count > THRESHOLD*total_pixels:
            rotation_vector.append(1)
        else:
            rotation_vector.append(0)
    
    return rotation_vector


def send_commands(rotation_vector, translation_x):
    """
    Send the commands to the drone based on the rotation and translation

    Args:
    rotation_vector
    translation_x

    """

    global curve

    ## Translation
    left_right = (translation_x - FRAME_WIDTH // 2) // SENSITIVITY
    left_right = int(np.clip(left_right, -10, 10)) # clip the speed

    if left_right < 2 and left_right > -2: 
        left_right = 0
    

    ## Rotation
    if rotation_vector == [1, 0, 0]: curve = WEIGHTS[0]
    elif rotation_vector == [1, 1, 0]: curve = WEIGHTS[1]
    elif rotation_vector == [0, 1, 0]: curve = WEIGHTS[2]
    elif rotation_vector == [0, 1, 1]: curve = WEIGHTS[3]
    elif rotation_vector == [0, 0, 1]: curve = WEIGHTS[4]

    elif rotation_vector == [0, 0, 0]: curve = WEIGHTS[2]
    elif rotation_vector == [1, 1, 1]: curve = WEIGHTS[2]
    elif rotation_vector == [1, 0, 1]: curve = WEIGHTS[2]


    # drone.send_rc_control(left_right, FORWARD_SPEED, 0, curve)


cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    
    imgThres = thresholding(img)
    translation_x = get_translation(imgThres, img)
    rotation_vector = get_rotation(imgThres, SENSITIVITY)

    cv.imshow("Output", imgThres)
    cv.waitKey(1)
