from torch.utils.data.dataloader import DataLoader
import torchvision
from customDataset import FloorPlanDataset
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import get_loaders, check_accuracy, save_checkpoint
from model import UNet
import matplotlib.pyplot as plt


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/sample_train/"
TRAIN_MASK_DIR = "data/sample_mask/"
VAL_IMG_DIR = "data/sample_train/"
VAL_MASK_DIR = "data/sample_mask/"


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

    test_ds = FloorPlanDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)
    checkpoint = torch.load('my_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    for i, (data, targets) in enumerate(test_loader):

        data = data.to(device=DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(data))
            preds = (preds > 0.5).float()


        torchvision.utils.save_image(
            preds, f'./saved_images/pred_{i}.jpg'
        )

        if i == 1:
            break


if __name__ == "__main__":
    predict()
