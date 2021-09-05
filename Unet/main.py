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

TRAIN_DIR = 'data/sample_train'
LABEL_DIR = 'data/sample_mask'
FILE_NAMES = os.listdir('./data/sample_train')

floorplan_dataset = FloorPlanDataset(TRAIN_DIR, LABEL_DIR, FILE_NAMES, transform=transforms.Compose([Rescale((512,512)), ToTensor()]))
# dataloader = DataLoader(floorplan_dataset, batch_size=1, shuffle=False)
dataloader = floorplan_dataset


#################
# TRAINING LOOP #
#################

model = Unet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

TRAIN = True

num_epochs = 20
n_total_steps = len(dataloader)
if TRAIN:
    for epoch in range(num_epochs):
        for i, item in enumerate(dataloader):

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

PATH = './cnn_sample.pth'
torch.save(model.state_dict(), PATH)


##############
# PREDICTION #
##############

PREDICT = False

if PREDICT:
    model.load_state_dict(torch.load(PATH))

    for i, item in enumerate(floorplan_dataset):
        output = model(item['train'])
        np_output = output.detach().numpy()[0, :, :, :]
        np_output = np_output.transpose((1,2,0))

        print(np_output.shape)

        sample_image(np_output)

        if i == 5:
            break
else:
    print('skipped prediction.')
    
