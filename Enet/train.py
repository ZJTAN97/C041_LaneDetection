import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from config.dataset import LaneDetectionDataset
from tqdm import tqdm
from imutils import paths
from config import config
from sklearn.model_selection import train_test_split
from model.ENet import ENet
from config.utils import (
    transforms,
    count_parameters,
    LRScheduler,
    EarlyStopping,
)
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_enet():

    image_paths = sorted(list(paths.list_images(config.IMAGE_DATATSET_PATH)))
    mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    split = train_test_split(
        image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42
    )

    (train_images, test_images) = split[:2]
    (train_masks, test_masks) = split[2:]

    train_ds = LaneDetectionDataset(
        image_paths=train_images, mask_paths=train_masks, transforms=transforms
    )
    test_ds = LaneDetectionDataset(
        image_paths=test_images, mask_paths=test_masks, transforms=transforms
    )

    print(f"[INFO] found {len(train_ds)} images in the training set...")
    print(f"[INFO] found {len(test_ds)} images in the validation set...")

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=config.BATCH_SIZE
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=config.BATCH_SIZE
    )

    enet = ENet(1)
    count_parameters(enet)
    pytorch_total_params = sum(
        p.numel() for p in enet.parameters() if p.requires_grad
    )
    print("num of parammeters: ", pytorch_total_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enet = enet.to(device)

    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(
        enet.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY
    )

    train_steps = len(train_ds) // config.BATCH_SIZE
    test_steps = len(test_ds) // config.BATCH_SIZE
    history = {"train_loss": [], "val_loss": []}

    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    # Training loop
    print("[INFO] training using E-Net architecture...")

    for e in tqdm(range(config.NUM_EPOCHS)):
        # set enet in training mode
        enet.train()

        total_train_loss = 0
        total_test_loss = 0

        for (i, (x, y)) in enumerate(train_loader):
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            pred = enet(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss

        with torch.no_grad():
            enet.eval()

            for (x, y) in test_loader:
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                pred = enet(x)
                total_test_loss += loss_fn(pred, y)

        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps

        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["val_loss"].append(avg_test_loss.cpu().detach().numpy())

        lr_scheduler(avg_test_loss)
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            break

        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print(
            "Train loss: {:.6f}, Validation loss: {:.6f}".format(
                avg_train_loss, avg_test_loss
            )
        )

        torch.save(
            {
                "state_dict": enet.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            config.MODEL_PATH,
        )

        print("saved weights successfully!")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.title("Training loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(config.PLOT_PATH)


if __name__ == "__main__":
    train_enet()
