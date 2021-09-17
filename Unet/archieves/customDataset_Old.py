import os
import torch
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2 as cv


class FloorPlanDataset(Dataset):
    def __init__(self, training_dir, label_dir, file_names, transform=None):
        self.training_dir = training_dir
        self.label_dir = label_dir
        self.file_names = file_names
        self.transform = transform
    

    def __len__(self):
        return len(self.file_names)
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        train_path = os.path.join(self.training_dir, self.file_names[index])
        label_path = os.path.join(self.label_dir, self.file_names[index])
        train_image = imread(train_path)
        label_image = imread(label_path, as_gray=True)
        label_image = label_image[:, :, None]

        sample = {
            'train': train_image,
            'label': label_image,
            }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        train, label = sample['train'], sample['label']

        h, w = train.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        train_resized = resize(train, (new_h, new_w), mode='constant', preserve_range=True)
        label_resized = resize(label, (new_h, new_w), mode='constant', preserve_range=True)

        return {'train': train_resized, 'label': label_resized}


class ToTensor(object):
    
    def __call__(self, sample):
        image, label = sample['train'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        # Research on DataLoader to do batch sizing.
        # image = np.expand_dims(image, axis=0)
        # label = np.expand_dims(label, axis=0)

        return {'train': torch.from_numpy(image),
                'label': torch.from_numpy(label)}




