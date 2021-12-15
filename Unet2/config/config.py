import torch
import os

ROOT_DIR = "../dataset"
TRAIN_PATH = "train_imgs"
MASK_PATH = "train_masks"

IMAGE_DATATSET_PATH = os.path.join(ROOT_DIR, TRAIN_PATH)
MASK_DATASET_PATH = os.path.join(ROOT_DIR, MASK_PATH)

TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False


# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.003
NUM_EPOCHS = 10
BATCH_SIZE = 1


# threshold to filter weak predictions
THRESHOLD = 0.5

BASE_OUTPUT = "output"


# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_weights.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
