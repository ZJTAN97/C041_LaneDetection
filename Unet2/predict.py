from config import config
import numpy as np
import torch
import cv2 as cv
import os


def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        image = cv.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        orig = image.copy()

        file_name = image_path.split(os.path.sep)[-1]
        ground_truth_path = os.path.join(config.MASK_DATASET_PATH, file_name)

        ground_truth_mask = cv.imread(ground_truth_path, 0)
        ground_truth_mask = cv.resize(ground_truth_mask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))


        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)

        ground_truth_mask_colored = cv.cvtColor(ground_truth_mask, cv.COLOR_GRAY2RGB)
        pred_mask_colored = cv.cvtColor(pred_mask, cv.COLOR_GRAY2RGB)
        stacked = np.concatenate((orig, ground_truth_mask_colored, pred_mask_colored), axis=1)
        cv.imshow('pred_mask', stacked)
        cv.waitKey(0)



print("[INFO] loading up test image paths...")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=2)

print("[INFO] loading model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

for path in image_paths:
    make_predictions(unet, path)


