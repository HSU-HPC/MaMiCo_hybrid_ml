import numpy as np
import torch
from torch.utils.data import Dataset


class MyFlowDataset(Dataset):
    def __init__(self, my_images, my_masks):
        self.sample_images = my_images
        self.sample_masks = my_masks

    def __len__(self):
        # assert len(self.sample_masks) == len(
        #    self.sample_images), "Image-Mask Dimension Mismatch."
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask
