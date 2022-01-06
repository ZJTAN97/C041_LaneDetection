from config import config
import numpy as np
import torch
import cv2 as cv
import os
from djitellopy import tello
import sys
from pathlib import Path
import keyboard

myDir = os.getcwd()
sys.path.append(myDir)
path = Path(myDir)
a = str(path.parent.absolute())
sys.path.append(a)

from Enet.model.ENet import ENet


FRAME_WIDTH = 256
FRAME_HEIGHT = 256
THRESHOLD = 0.2

SENSITIVITY = 3
WEIGHTS = [-25, -15, 0, 15, 15]
FORWARD_SPEED = 15
CURVE = 0


def get_rotation(img, sensors):
    """
    Args:
    processedImg --> Preprocesed image
    sensors --> how sensitive the drone will be in terms of rotation

    Returns:
    rotation_vector of the drone
    """

    img_split = np.hsplit(img, 2)
    total_pixels = (img.shape[1] // sensors) * img.shape[0]
    rotation_vector = []

    for img in img_split:
        pixel_count = cv.countNonZero(img)
        if pixel_count > THRESHOLD * total_pixels:
            rotation_vector.append(1)
        else:
            rotation_vector.append(0)

    return rotation_vector


def get_translation(predictions, img):
    """
    To get the contours from the prediction of a trained model
    Contours will handle the translation motion of the drone
    """
    cx = 0
    contours, hierarchy = cv.findContours(
        predictions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    if len(contours) != 0:
        # biggest = max(contours, key=cv.contourArea)
        # x, y, w, h = cv.boundingRect(biggest)

        biggestContours = sorted(contours, key=cv.contourArea)[-2:]

        if len(biggestContours) > 1:

            x1, y1, w1, h1 = cv.boundingRect(biggestContours[0])
            x2, y2, w2, h2 = cv.boundingRect(biggestContours[1])

            # center of x and y
            cx = x1 + x2 // 2
            cy = y1 + y2 // 2

            cv.drawContours(img, biggestContours, -1, (0, 255, 0), 7)
            cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)

    return cx


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
    left_right = int(np.clip(left_right, -10, 10))  # clip the speed

    print(left_right)

    if left_right < 2 and left_right > -2:
        left_right = 0

    ## Rotation
    if rotation_vector == [1, 0, 0]:
        curve = WEIGHTS[0]
    elif rotation_vector == [1, 1, 0]:
        curve = WEIGHTS[1]
    elif rotation_vector == [0, 1, 0]:
        curve = WEIGHTS[2]
    elif rotation_vector == [0, 1, 1]:
        curve = WEIGHTS[3]
    elif rotation_vector == [0, 0, 1]:
        curve = WEIGHTS[4]

    elif rotation_vector == [0, 0, 0]:
        curve = WEIGHTS[2]
    elif rotation_vector == [1, 1, 1]:
        curve = WEIGHTS[2]
    elif rotation_vector == [1, 0, 1]:
        curve = WEIGHTS[2]

    # drone.send_rc_control(left_right, FORWARD_SPEED, 0, curve)


def main():

    model = ENet(1)
    checkpoint = torch.load(
        config.PRE_TRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["state_dict"])

    print("[INFO] Model Loaded...")
    print("Initializing Flight...")

    drone = tello.Tello()
    drone.connect()
    print("--- Drone Connected ---")
    print(f"-- Drone Battery {drone.get_battery()}% ---")
    drone.streamon()

    while True:

        # if keyboard.is_pressed("f"):
        #     print('taking off...')
        #     drone.takeoff()

        if keyboard.is_pressed("e"):
            drone.land()
            print("[INFO] Emergency Landing...")

        # manual override
        if keyboard.is_pressed("w"):
            drone.send_rc_control(0, 60, 0, 0)
        if keyboard.is_pressed("s"):
            drone.send_rc_control(0, -60, 0, 0)
        if keyboard.is_pressed("a"):
            drone.send_rc_control(-60, 0, 0, 0)
        if keyboard.is_pressed("d"):
            drone.send_rc_control(60, 0, 0, 0)

        with torch.no_grad():

            frame = drone.get_frame_read().frame
            frame = cv.flip(
                frame, 0
            )  # uncomment when using drone / drone's footages
            frame = cv.resize(
                frame, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
            )
            orig = frame.copy()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = frame.astype("float32") / 255.0

            frame = np.transpose(frame, (2, 0, 1))
            frame = np.expand_dims(frame, 0)
            frame = torch.from_numpy(frame).to(config.DEVICE)

            pred = model(frame).squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()

            pred = (pred > config.THRESHOLD) * 255
            pred = pred.astype(np.uint8)

            translation_x = get_translation(pred, orig)
            # rotation = get_rotation(pred, 3)

            send_commands([0, 0, 0], translation_x)

            cv.imshow("output", orig)
            # cv.imshow("prediction", pred)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
