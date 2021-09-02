import os
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


# Sanity Check
# sample_data = torch.stack((floorplan_dataset[0]['train'], floorplan_dataset[0]['train']), 0)
# y = model(sample_data)
# print(y.shape)



criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
n_total_steps = len(floorplan_dataset)

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


