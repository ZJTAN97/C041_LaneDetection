from config.dataset import LaneDetectionDataset
from config.model import UNet
from config.dice_loss import DiceBCELoss
from config import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time


image_paths = sorted(list(paths.list_images(config.IMAGE_DATATSET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

split = train_test_split(
    image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42
)

(train_images, test_images) = split[:2]
(train_masks, test_masks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(test_images))
f.close()


# Data Loading Pipeline
transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
        ),
        transforms.ToTensor(),
    ]
)

train_ds = LaneDetectionDataset(
    image_paths=train_images, mask_paths=train_masks, transforms=transforms
)
test_ds = LaneDetectionDataset(
    image_paths=test_images, mask_paths=test_masks, transforms=transforms
)

print(f"[INFO] found {len(train_ds)} examples in the training set...")
print(f"[INFO] found {len(test_ds)} examples in the test set...")

train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=config.BATCH_SIZE)


# Initialize UNet Model
unet = UNet().to(config.DEVICE)
loss_fn = BCEWithLogitsLoss()  # implement diceloss later
dice_loss_fn = DiceBCELoss()
optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

train_steps = len(train_ds) // config.BATCH_SIZE
test_steps = len(test_ds) // config.BATCH_SIZE
history = {"train_loss": [], "test_loss": []}


# Training Loop
print("[INFO] training the network...")
start_time = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    # set model in training mode
    unet.train()

    total_train_loss = 0
    total_test_loss = 0

    for (i, (x, y)) in enumerate(train_loader):
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        pred = unet(x)
        loss = dice_loss_fn(pred, y)

        # zero any previously accumulated gradients
        # perform backpropagation and update modal parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss

    # switch off autograd
    with torch.no_grad():
        # set model in evaluation mode
        unet.eval()

        for (x, y) in test_loader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = unet(x)
            total_test_loss += dice_loss_fn(pred, y)

    avg_train_loss = total_train_loss / train_steps
    avg_test_loss = total_test_loss / test_steps

    history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    history["test_loss"].append(avg_test_loss.cpu().detach().numpy())

    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print(
        "Train loss: {:.6f}, Test loss: {:.6f}".format(
            avg_train_loss, avg_test_loss
        )
    )
    torch.save(
        {"state_dict": unet.state_dict(), "optimizer": optimizer.state_dict()},
        config.MODEL_PATH,
    )
    print("saved weights successfully!")

end_time = time.time()
print(
    "[INFO] total time taken to trian the model: {:.2f}s".format(
        end_time - start_time
    )
)


plt.style.use("ggplot")
plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["test_loss"], label="test_loss")
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
plt.savefig(config.PLOT_PATH)
