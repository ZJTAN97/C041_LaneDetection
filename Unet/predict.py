from torch.utils.data.dataloader import DataLoader
import torchvision
from custom_dataset import LaneTestSet
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


TEST_IMG_DIR = "test_imgs/"


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

    test_ds = LaneTestSet(
        image_dir=TEST_IMG_DIR,
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

    for i, data in enumerate(test_loader):
        data = data.to(device=DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(data))
            preds = (preds > 0.5).float()


        torchvision.utils.save_image(
            preds, f'./predicted_masks/pred_{i}.jpg'
        )


if __name__ == "__main__":
    predict()
