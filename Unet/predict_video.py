from config import config
from config.model import UNet
import numpy as np
import torch
import cv2 as cv


def make_predictions_video(video_path):

    model = UNet()
    checkpoint = torch.load(
        config.PRE_TRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["state_dict"])

    cap = cv.VideoCapture(video_path)

    while True:

        with torch.no_grad():
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

            cv.imshow("Comparison", stacked)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break


path = "../dataset/test_videos/test_video_1fps.mp4"
make_predictions_video(path)
