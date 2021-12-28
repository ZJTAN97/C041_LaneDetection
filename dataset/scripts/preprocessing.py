import scipy.io
import cv2 as cv
import numpy as np
import argparse


def preprocessing():

    parser = argparse.ArgumentParser(
        description="Please indicate your MATLAB matrix file path. If --exist count not declared, will be taken as 0."
    )
    parser.add_argument("--mat")
    parser.add_argument("--exist")
    args = parser.parse_args()

    if args.mat == None:
        raise FileNotFoundError(
            "File path does not exist. Include --mat arg to indicate file path"
        )
    if args.exist == None:
        raise IndexError("Please state index")

    MATLAB_DIR = "../matlab_labels"
    TRAIN_DIR = "../train_imgs"
    MASK_DIR = "../train_masks"
    MATLAB_MATRIX = args.mat
    EXIST = int(args.exist)

    mat = scipy.io.loadmat(f"{MATLAB_DIR}/{MATLAB_MATRIX}")
    labels = list(mat.values())[3:]

    for i, img in enumerate(labels):
        threshold, thresh = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
        gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv.dilate(gray, kernel, iterations=2)
        cv.imwrite(f"{MASK_DIR}/image_{i + EXIST}.jpg", dilate)


if __name__ == "__main__":
    preprocessing()
