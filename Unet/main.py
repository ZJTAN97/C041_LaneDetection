import os
import torch
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

sample_data = torch.stack((floorplan_dataset[0]['train'], floorplan_dataset[0]['train']), 0)


y = model(sample_data)
print(y.shape)