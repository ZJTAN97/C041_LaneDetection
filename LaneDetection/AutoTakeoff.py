from config import config
import numpy as np
import torch
import cv2 as cv
import os
from djitellopy import tello
import sys
from pathlib import Path

myDir = os.getcwd()
sys.path.append(myDir)
path = Path(myDir)
a = str(path.parent.absolute())
sys.path.append(a)

from Enet.model.ENet import ENet


def auto_takeoff(predictions, img, drone):
    """
    To get the contours from the prediction of a trained model
    Contours will handle the translation motion of the drone
    """
    cx = 0
    contours, hierarchy = cv.findContours(
        predictions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if len(contours) != 0:

        biggestContours = sorted(contours, key=cv.contourArea)[-2:]

        for c in biggestContours:
            M = cv.moments(c)
            cX = int(M["m10"] / (M["m00"] if M["m00"] != 0 else 1))
            cY = int(M["m01"] / (M["m00"] if M["m00"] != 0 else 1))
            # cv.circle(img, (cX, cY), 5, (0, 0, 255), -1)

            print(cY)

            if cY > 210:
                print("takeoff!")
                # drone.takeoff()

        cv.fillPoly(img, contours, color=(0, 255, 0))

        # cv.polylines(img, biggestContours, True, (0,255,0), 2)
        # cv.fillPoly(img, contours, color=(0,255,0))


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

    # cap = cv.VideoCapture("../dataset/test_videos/takeoff.mp4")

    while True:

        with torch.no_grad():

            # success, frame = cap.read()

            frame = drone.get_frame_read().frame
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

            auto_takeoff(pred, orig, drone)

            cv.imshow("output", orig)
            # cv.imshow("prediction", pred)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    drone.land()


if __name__ == "__main__":
    main()
