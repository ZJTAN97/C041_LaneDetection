from config import config
import numpy as np
import torch
import cv2 as cv
import os
import sys
from pathlib import Path

myDir = os.getcwd()
sys.path.append(myDir)
path = Path(myDir)
a = str(path.parent.absolute())
sys.path.append(a)

from Enet.model.ENet import ENet

## Re train in 255..
## use 240 if u want test rotation

FRAME_WIDTH = 256
FRAME_HEIGHT = 256
THRESHOLD = 0.2

SENSITIVITY = 3
WEIGHTS = [-25, -15, 0, 15, 15]
FORWARD_SPEED = 15
CURVE = 0
sensors = 3


def get_rotation(predictions, sensors):

    img = np.hsplit(predictions, sensors)  # split img into 3

    for idx, im in enumerate(img):
        cv.imshow(str(idx), im)


def get_translation(predictions, img):

    contours, hierarchy = cv.findContours(
        predictions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    if len(contours) != 0:
        # biggest = max(contours, key=cv.contourArea)
        # x, y, w, h = cv.boundingRect(biggest)

        biggestContours = sorted(contours, key=cv.contourArea)[-2:]

        x1, y1, w1, h1 = cv.boundingRect(biggestContours[0])
        x2, y2, w2, h2 = cv.boundingRect(biggestContours[1])

        # center of x and y
        cx = x1 + x2 // 2
        cy = y1 + y2 // 2

        cv.drawContours(img, biggestContours, -1, (0, 255, 0), 7)
        cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)

    return cx


cap = cv.VideoCapture(0)
model = ENet(1)
checkpoint = torch.load(
    config.PRE_TRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["state_dict"])


while True:
    _, frame = cap.read()
    frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    with torch.no_grad():
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
    # rotation = get_rotation(pred, sensors=sensors)

    cv.imshow("output", orig)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
