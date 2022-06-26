import numpy as np
import torch
from torch.utils.data import Dataset


class MyMamicoDataset(Dataset):
    #
    # BRIEF: This class inherits from the torch Dataset class and allows
    # to create a userdefined dataset. Here, the dataset is hardcoded
    # to consider md+outer dimensionality of 26^3 consisting of:
    # 1xghost_cell, 3xouter_cell, 18xmd_cell, 3xouter_cell, 1xghost_cell
    # This dataset then takes the 26^3 input and removes the ghost_cells
    # for the inputs(=images) and further removes the outer_cells for
    # the outputs (=masks).
    # Note that the data dimensions represent:
    # timestep, RGB-channel (=x/y/z velocity), x-pos, y-pos, z-pos
    #
    def __init__(self, my_images):
        self.sample_images = my_images[:-1, :, 1:-1, 1:-1, 1:-1]
        # print("Dataset.py - Sanity Check - Shape of sample_images: ", self.sample_images.shape)
        self.sample_masks = my_images[1:, :, 4:22, 4:22, 4:22]
        # print("Dataset.py - Sanity Check - Shape of sample_masks: ", self.sample_masks.shape)

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


class MyMamicoDataset_UNET_AE(Dataset):
    # BRIEF: This class inherits from the torch Dataset class and allows
    # to create a userdefined dataset. Here, the dataset is hardcoded
    # to consider md+outer dimensionality of 26^3 consisting of:
    # 1xghost_cell, 3xouter_cell, 18xmd_cell, 3xouter_cell, 1xghost_cell
    # As this Dataset will be used for the autoencoder U-Net training,
    # this dataset takes the 26^3 input and removes the ghost_cells.
    # The inputs(=images) and outputs are therefor identical.
    #
    # Note that the data dimensions represent:
    # timestep, RGB-channel (=x/y/z velocity), x-pos, y-pos, z-pos
    #
    def __init__(self, my_images):
        self.sample_images = my_images[:, :, 1:-1, 1:-1, 1:-1]
        print("Dataset.py - Sanity Check - Shape of sample_images: ",
              self.sample_images.shape)
        self.sample_masks = my_images[:, :, 1:-1, 1:-1, 1:-1]
        print("Dataset.py - Sanity Check - Shape of sample_masks: ",
              self.sample_masks.shape)
        print('Both should be 1000 x 3 x 24 x 24 x 24')

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


class MyMamicoDataset_RNN(Dataset):
    #
    # This class inherits from the torch Dataset class and allows
    # to create a userdefined dataset. Here, the dataset is hardcoded
    # to consider md+outer dimensionality of 26^3 consisting of:
    # 1xghost_cell, 3xouter_cell, 18xmd_cell, 3xouter_cell, 1xghost_cell
    # This dataset then takes the 26^3 input and removes the ghost_cells
    # for the inputs(=images) and further removes the outer_cells for
    # the outputs (=masks).
    # Note that the data dimensions represent:
    # timestep, RGB-channel (=x/y/z velocity), x-pos, y-pos, z-pos
    #
    def __init__(self, my_images, seq_length=15):
        self.sample_masks = my_images[seq_length:]
        # print("Dataset.py - Sanity Check - Shape of sample_masks: ", self.sample_masks.shape)
        self.sample_images = np.zeros((
            len(my_images)-seq_length-1, seq_length, 256))

        for i in range(len(self.sample_images)):
            self.sample_images[i] = my_images[i:seq_length+i]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask
