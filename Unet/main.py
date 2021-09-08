import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
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

TRAIN_DIR = 'data/sample_train/'
LABEL_DIR = 'data/sample_mask/'
FILE_NAMES = os.listdir('./data/sample_train')


floorplan_dataset = FloorPlanDataset(TRAIN_DIR, LABEL_DIR, FILE_NAMES, transform=transforms.Compose([Rescale((512,512)), ToTensor()]))
dataloader = DataLoader(floorplan_dataset, batch_size=1, shuffle=False)


#################
# TRAINING LOOP #
#################

model = Unet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

TRAIN = True
EPOCHS = 50

if TRAIN:
    for epoch in range(EPOCHS):
        for i, item in enumerate(dataloader):

            # Forward Pass
            predictions = model(item['train'])
            loss = criterion(predictions, item['label'])

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
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

    for i, item in enumerate(dataloader):
        output = model(item['train'])
        thres = torch.tensor([0.8])
        output = (output > thres).float() * 1

        print(output)
        print(output.shape)

        np_output = output.detach().numpy()[0, :, :, :]
        np_output = np_output.transpose((1,2,0))


        plt.imshow(np_output)
        plt.show()

        if i == 1:
            break
else:
    print('skipped prediction.')
    
