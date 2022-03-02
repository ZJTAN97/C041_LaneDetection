from config import config
from model.ENet import ENet
import numpy as np
import torch
import cv2 as cv
import os
import time


def form_rectangle(predictions, img):
    """
    To get the contours from the prediction of a trained model
    Contours will handle the translation motion of the drone
    """
    contours, hierarchy = cv.findContours(
        predictions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if len(contours) != 0:

        biggestContours = sorted(contours, key=cv.contourArea)[
            -2:
        ]  # this will get 2 lanes
        x1, y1, w1, h1 = cv.boundingRect(biggestContours[0])
        x2, y2, w2, h2 = cv.boundingRect(biggestContours[1])

        # center of x and y
        cx = (x1 + x2) // 2
        cy = h1

        # top top, bottom bottom
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


def make_predictions(image_path):

    model = ENet(1)
    checkpoint = torch.load(
        config.PRE_TRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["state_dict"])

    with torch.no_grad():
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        image = cv.resize(
            image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        )
        original = image.copy()

        file_name = image_path.split(os.path.sep)[-1]
        ground_truth_path = os.path.join(config.MASK_DATASET_PATH, file_name)

        ground_truth_mask = cv.imread(ground_truth_path, 0)
        ground_truth_mask = cv.resize(
            ground_truth_mask,
            (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT),
        )

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)

        ground_truth_mask_colored = cv.cvtColor(
            ground_truth_mask, cv.COLOR_GRAY2RGB
        )
        pred_mask_colored = cv.cvtColor(pred_mask, cv.COLOR_GRAY2RGB)
        stacked = np.concatenate(
            (original, ground_truth_mask_colored, pred_mask_colored), axis=1
        )

        contours, hierarchy = cv.findContours(
            pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if len(contours) != 0:

            biggestContours = sorted(contours, key=cv.contourArea)[
                -2:
            ]  # this will get 2 lanes
            x1, y1, w1, h1 = cv.boundingRect(biggestContours[0])
            x2, y2, w2, h2 = cv.boundingRect(biggestContours[1])

            # center of x and y
            cx = (x1 + x2) // 2
            cy = h1

            # top top, bottom bottom
            boundingRect = np.array(
                [
                    [x2, y2],
                    [x1, y1],
                    [x1, y1 + h1],
                    [x2, y2 + h2],
                ]
            )

            cv.drawContours(original, [boundingRect], -1, (0, 255, 0), 2)
            cv.circle(original, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        cv.imshow("pred_mask", stacked)
        cv.waitKey(0)


print("[INFO] loading up test image paths....")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=10)

for path in image_paths:
    make_predictions(path)


def make_predictions_video(video_path):

    model = ENet(1)
    checkpoint = torch.load(
        config.PRE_TRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["state_dict"])

    cap = cv.VideoCapture(video_path)

    while True:

        with torch.no_grad():

            start_time = time.time()
            success, frame = cap.read()
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

            pred_colored = cv.cvtColor(pred, cv.COLOR_GRAY2RGB)
            stacked = np.concatenate((orig, pred_colored), axis=1)

            end_time = time.time()
            print(
                "Time taken for frame prediction: {:.2f}s".format(
                    end_time - start_time
                )
            )

            cv.imshow("Comparison", stacked)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break


# path = "../dataset/test_videos/test_video_2.mp4"
# make_predictions_video(path)
