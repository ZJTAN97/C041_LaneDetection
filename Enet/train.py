import torch
from torch.nn import BCEWithLogitsLoss
from model.ENet import ENet
from config.utils import loader
import os
from tqdm import tqdm

enet = ENet(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enet = enet.to(device)


ROOT_DIR = "../dataset"
TRAIN_PATH = "train_imgs"
MASK_PATH = "train_masks"

IMAGE_DATATSET_PATH = os.path.join(ROOT_DIR, TRAIN_PATH)
MASK_DATASET_PATH = os.path.join(ROOT_DIR, MASK_PATH)

train = loader("../dataset/train_imgs/", "../dataset/train_masks/", 1)
epochs = 100

criterion = BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(enet.parameters(), lr=5e-4, weight_decay=2e-4)

print_every = 5
eval_every = 5

for e in range(1, epochs + 1):
    train_loss = 0
    print("-" * 15, "Epoch %d" % e, "-" * 15)

    enet.train()

    for _ in tqdm(range(82)):
        X_batch, mask_batch = next(train)

        X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()

        out = enet(X_batch.float())

        loss = criterion(out, mask_batch.long())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
