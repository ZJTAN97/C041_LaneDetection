import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet
import cv2 as cv
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


def predict():

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)
    checkpoint = torch.load('my_checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    cap = cv.VideoCapture('test_video_1fps.mp4')

    i = 0

    while True:
        success, frame = cap.read()
        # frame_resized = cv.resize(frame, (256,256)).astype(np.float32)
        frame = cv.resize(frame, (256,256))
        frame = frame.astype(np.float32)
        frame /= 255.



        # if i % 30 == 0:
        augmentations = test_transforms(image=frame)
        image = augmentations['image']
        with torch.no_grad():
            preds = torch.sigmoid(model(image.unsqueeze(0)))
            preds = (preds > 0.5).float()

        prediction = preds.squeeze(0)
        prediction = prediction.numpy()
        prediction = np.swapaxes(prediction, 0, 1)
        prediction = np.swapaxes(prediction, 1, 2)
        prediction = cv.cvtColor(prediction, cv.COLOR_GRAY2BGR)


        result = cv.addWeighted(prediction, 1, frame, 1, 0)


        # i += 1        

        cv.imshow('Video', result)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    predict()
