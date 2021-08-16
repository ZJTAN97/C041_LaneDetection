import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.utils.data

import cv2 as cv

from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.facecolor'] = '#ffffff' 



########################
# IMAGE PRE-PROCESSING #
########################

IMG_WIDTH = 768 
IMG_HEIGHT = 768 
IMG_CHANNELS = 3
TRAIN_PATH = './data/train/'
TRAIN_PATH_MASK = './data/mask/' 
TEST_PATH = './data/test/' 
SEED = 42

np.random.seed = SEED
train_ids = next(os.walk(TRAIN_PATH))
train_ids = train_ids[2]

train_mask_ids = next(os.walk(TRAIN_PATH_MASK)) 
train_mask_ids = train_mask_ids[2]

test_ids = next(os.walk(TEST_PATH)) 
test_ids = test_ids[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) 

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


print('Resizing training images and storing it to X_train')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    #read mask and image
    img = cv.imread(path)[:,:,:IMG_CHANNELS]
    img = cv.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X_train[n] = img  #Fill empty X_train with values from img

print('Resizing mask images and storing it to Y_train')
for n, id_ in tqdm(enumerate(train_mask_ids), total=len(train_mask_ids)):
    path = TRAIN_PATH_MASK + id_
    mask_ = cv.imread(path)
    mask_ = cv.cvtColor(mask_, cv.COLOR_BGR2GRAY)
    mask_ = np.expand_dims(cv.resize(mask_, (IMG_HEIGHT, IMG_WIDTH)), axis=-1)
    Y_train[n] = mask_ #Fill empty Y_train with values from mask

sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = cv.imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = cv.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X_test[n] = img
print("resizing for all images completed")