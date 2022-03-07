import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

INPUT_IMAGE_HEIGHT = 240
INPUT_IMAGE_WIDTH = 240

THRESHOLD = 0.4
BASE_OUTPUT = "weights"
PRE_TRAINED_WEIGHTS_PATH = os.path.join(BASE_OUTPUT, "enet_130_100.pth.tar")
