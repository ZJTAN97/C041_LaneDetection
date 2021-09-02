import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from utils import sample_image
from model import Unet
from customDataset import FloorPlanDataset, Rescale, ToTensor


########################
# IMAGE PRE-PROCESSING #
########################

TRAIN_DIR = 'data/train'
LABEL_DIR = 'data/mask'
FILE_NAMES = os.listdir('./data/train')

floorplan_dataset = FloorPlanDataset(TRAIN_DIR, LABEL_DIR, FILE_NAMES, transform=transforms.Compose([Rescale((512,512)), ToTensor()]))

model = Unet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

TRAIN = False

num_epochs = 10
n_total_steps = len(floorplan_dataset)
if TRAIN:
    for epoch in range(num_epochs):
        for i, item in enumerate(floorplan_dataset):

            # Forward Pass
            predictions = model(item['train'])
            loss = criterion(predictions, item['label'])

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1} / {num_epochs}, Step {i+1} / {len(floorplan_dataset)}, Loss: {loss.item():.4f}')
else:
    print('Loading trained model..')



PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH))

for i, item in enumerate(floorplan_dataset):
    output = model(item['train'])
    np_output = output.detach().numpy()[0, :, :, :]
    np_output = np_output.transpose((1,2,0))

    print(np_output.shape)

    sample_image(np_output)

    if i == 5:
        break
    
