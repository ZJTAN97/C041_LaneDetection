from torch.utils.data import Dataset
import cv2 as cv


class LaneDetectionDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        """
        standard length method
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        getter method
        """
        image = cv.imread(self.image_paths[idx])
        # OpenCV by defaults load BGR, so we convert to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Load masks in grayscale
        mask = cv.imread(self.mask_paths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)
