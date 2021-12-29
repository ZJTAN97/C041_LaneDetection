# C041 - Lane Detection

This repository contains my work for my Final Year Project, C041 Lane Detection. The repository can be further
broken down into the following segments. The entire project is under constant updates and will change in the near future.

1. U Net
    - config: required configurations for training and prediction
    - model: the architecture of U Net written in PyTorch
    - output: consists of the loss plots and pre-trained weights
    - train.py: to start training using U Net architecture.
    - predict.py: to predict/test pre-trained weights using U Net architecture

2. E Net
    - config: required configurations for training and prediction
    - model: the architecture of U Net written in PyTorch
    - output: consists of the loss plots and pre-trained weights
    - train.py: to start training using E Net architecture.
    - predict.py: to predict/test pre-trained weights using E Net architecture

3. Lane Detection
    - currently in major development
    - consists of integration of DJI tellopy API and trained weights.
    - Responsible for drone lane detection

4. Dataset
    - matlab_labels
    - sample_mask
    - sample_train
    - scripts
    - test_videos
    - train_imgs
    - train_masks
    - unprocessed


# Getting Started

## To add new Images to dataset.
1. Use MATLAB Image segmenter
2. Draw out the polygon, apply binary on the segmenter, export the masked Image
3. Select all the maskedImaged, save as a matlab matrix `.mat`
4. Use preprocess.py to process the images.
