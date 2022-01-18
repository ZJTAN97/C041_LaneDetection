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

        biggestContours = sorted(contours, key=cv.contourArea)[
            -2:
        ]  # this will get 2 lanes
        x1, y1, w1, h1 = cv.boundingRect(biggestContours[0])  # right lane
        x2, y2, w2, h2 = cv.boundingRect(biggestContours[1])  # left lane

        # center of x and y
        cx = (x1 + x2) // 2
        cy = (h1) // 2

        boundingRect = np.array(
            [
                [x2, y2],
                [x1, y1],
                [x1, y1 + h1],
                [x2, y2 + h2],
            ]
        )

        cv.drawContours(img, [boundingRect], -1, (0, 255, 0), 2)
        cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

    return cx


def get_rotation(prediction):

    split_img = np.hsplit(prediction, 3)
    total_pixels = (prediction.shape[1] // 3) * prediction.shape[0]
    rotation_coordinates = []

    for (
        i,
        img,
    ) in enumerate(split_img):
        pixel_count = cv.countNonZero(img)
        if pixel_count > THRESHOLD * total_pixels:
            rotation_coordinates.append(1)
        else:
            rotation_coordinates.append(0)

    return rotation_coordinates


def send_commands(translation, rotation, drone):
    """
    Send the commands to the drone based on the rotation and translation

    Args:
    rotation_vector
    translation_x

    """
    ## Translation
    left_right = (translation - FRAME_WIDTH // 2) // SENSITIVITY
    left_right = int(np.clip(left_right, -10, 10))  # clip the speed

    if rotation == [1, 0, 0]:
        rotate = -25
    elif rotation == [1, 1, 0]:
        rotate = -10
    elif rotation == [0, 1, 1]:
        rotate = 10
    elif rotation == [0, 0, 1]:
        rotate = 25
    else:
        rotate = 0

    drone.send_rc_control(left_right, FORWARD_SPEED, 0, rotate)


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
    drone.streamoff()  # clear any existing streams
    drone.streamon()

    while True:

        if keyboard.is_pressed("f"):
            print("taking off...")
            drone.takeoff()

        if keyboard.is_pressed("e"):
            drone.land()
            print("[INFO] Emergency Landing...")

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
            rotation = get_rotation(pred)

            send_commands(translation_x, rotation, drone)

            cv.imshow("output", orig)
            # cv.imshow("prediction", pred)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    drone.land()


if __name__ == "__main__":
    main()
