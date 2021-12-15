import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import get_loaders, check_accuracy, save_checkpoint
from model import UNet
from dice_loss import DiceBCELoss

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 85
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

ROOT_DIR = "../dataset"

TRAIN_IMG_DIR = f"{ROOT_DIR}/train_imgs"
TRAIN_MASK_DIR = f"{ROOT_DIR}/train_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):

    for index, (data, targets) in enumerate(loader):

        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f'Step {index+1} / {len(loader)}')


def main():
    
    train_transforms = A.Compose(
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

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, dice_loss_fn, scaler)

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        })

        check_accuracy(val_loader, model, device=DEVICE)
        print(f'Epoch {epoch+1} / {NUM_EPOCHS}')
        


if __name__ == "__main__":
    main()