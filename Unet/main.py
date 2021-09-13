import os
import cv2 as cv
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.std import tqdm

from utils import get_loaders
from model2 import UNet
from customDataset import FloorPlanDataset


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


########################
# IMAGE PRE-PROCESSING #
########################

TRAIN_IMG_DIR = "data/sample_train/"
LABEL_IMG_DIR = "data/sample_mask/"
VAL_IMG_DIR = "data/sample_train/"
VAL_MASK_DIR = "data/sample_mask/"


#################
# TRAINING LOOP #
#################



def main(TRAIN):

    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        LABEL_IMG_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    EPOCHS = 50

    if TRAIN:
        for epoch in range(EPOCHS):
            for i, (data, targets) in enumerate(train_loader):

                data = data.to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)

                # Forward Pass
                predictions = model(data)
                loss = loss_fn(predictions, targets)

                # Backward and Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    else:
        print('Loading trained model..')

    PATH = './cnn_sample.pth'
    torch.save(model.state_dict(), PATH)


##############
# PREDICTION #
##############

# PREDICT = False

# if PREDICT:
#     model.load_state_dict(torch.load(PATH))

#     for i, item in enumerate(train_loader):
#         output = model(item['train'])
#         # thres = torch.tensor([0.8])
#         # output = (output > thres).float() * 1

#         # print(output)
#         # print(output.shape)

#         np_output = output.detach().numpy()[0, :, :, :]
#         np_output = np_output.transpose((1,2,0))


#         plt.imshow(np_output)
#         plt.show()

#         if i == 1:
#             break
# else:
#     print('skipped prediction.')
    

if __name__ == "__main__":
    main(True)