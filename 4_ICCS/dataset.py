import numpy as np
import torch
from torch.utils.data import Dataset


class MyMamicoDataset_AE(Dataset):
    """This class inherits from the torch Dataset class and allows to create a
    userdefined dataset. Here, the dataset is hardcoded to consider md+outer
    dimensionality of 26^3 consisting of:
    1x ghost_cell, 3x outer_cell, 18x md_cell, 3x outer_cell, 1x ghost_cell
    As this Dataset will be used for the autoencoder U-Net training, this
    dataset takes the 26^3 input and removes the ghost_cells. The inputs
    (=images) and outputs (=masks) are therefor identical.

    Note that the data dimensions [d_0, d_1, d_2, d_3, d_4] represent:
    d_0 = 1000 -> timestep,
    d_1 = 3    -> RGB-channel (= u_x, u_y, u_z),
    d_2 = 24   -> x-pos,
    d_3 = 24   -> y-pos,
    d_4 = 24   -> z-pos

    Args:
        my_images:
          Object of type numpy array containing the timeseries of multichannel
          volumetric data.
    """

    def __init__(self, my_images):
        self.sample_images = my_images[:, :, 1:-1, 1:-1, 1:-1]
        self.sample_masks = my_images[:, :, 1:-1, 1:-1, 1:-1]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


class MyMamicoDataset_RNN(Dataset):
    """This class inherits from the torch Dataset class and allows to create a
    userdefined dataset. Here, the dataset is hardcoded to consider the AE
    generated latent space of dimension [1000, 256] on the basis of MaMiCo
    generated MD+outer data of dimension [1000, 3, 24, 24, 24]. In particular,
    it creates datasets of dimension [d_0 - seq_length - 1, seq_length, 256]
    tailored to a specific seq_length and used to predict the next timestep's
    latentspace.

    Args:
        my_images:
          Object of type numpy array containing the timeseries of latentspaces.
        seq_length:
          Object of integer type specifying the number of elements to include
          in the RNN sequence.
    """

    def __init__(self, my_images, seq_length=15):
        self.sample_masks = my_images[seq_length:]
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


class MyMamicoDataset_RNN_verification(Dataset):
    """This class inherits from the torch Dataset class and allows to create a
    userdefined dataset. Here, the dataset is hardcoded to consider latentspaces
    of shape (1000, 256) derived from the md+outer dimensionality of 26^3.
    As this Dataset will be used to verify proper latent space generation, the
    inputs (=images) and outputs (=masks) are therefor identical.

    Args:
        my_images:
          Object of type numpy array containing the timeseries channel-specific
          latentspaces.
    """

    def __init__(self, my_images):
        self.sample_images = my_images[:, :]
        self.sample_masks = my_images[:, :]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


class MyMamicoDataset_Hybrid(Dataset):
    """This class inherits from the torch Dataset class and allows to create a
    userdefined dataset. Here, the dataset is hardcoded to consider md+outer
    dimensionality of 26^3 consisting of:
    1x ghost_cell, 3x outer_cell, 18x md_cell, 3x outer_cell, 1x ghost_cell
    As this dataset will be used for Hybrid_MD_RNN_UNET validation, this
    dataset first removes the ghost_cells and then creates input (=image) and
    output (=mask) pairs for timeseries prediction. In other words, the image
    is for timestep = t and the mask is for timestep = t+1.

    Note that the data dimensions [d_0, d_1, d_2, d_3, d_4] represent:
    d_0 = 999  -> timestep,
    d_1 = 3    -> RGB-channel (= u_x, u_y, u_z),
    d_2 = 24   -> x-pos,
    d_3 = 24   -> y-pos,
    d_4 = 24   -> z-pos

    Args:
        my_images:
          Object of type numpy array containing the timeseries of multichannel
          volumetric data.
    """

    def __init__(self, my_images):
        self.sample_images = my_images[:-1, :, 1:-1, 1:-1, 1:-1]
        self.sample_masks = my_images[1:, :, 1:-1, 1:-1, 1:-1]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


class MyMamicoDataset_Hybrid_analysis(Dataset):
    """This class inherits from the torch Dataset class and allows to create a
    userdefined dataset. Here, the dataset is hardcoded to consider md+outer
    dimensionality of 26^3 consisting of:
    1x ghost_cell, 3x outer_cell, 18x md_cell, 3x outer_cell, 1x ghost_cell
    As this dataset will be used for Hybrid_MD_RNN_UNET validation, this
    dataset first removes the ghost_cells and then creates input (=image) and
    output (=mask) pairs for timeseries prediction. In other words, the image
    is for timestep = t and the mask is for timestep = t+20.

    Note that the data dimensions [d_0, d_1, d_2, d_3, d_4] represent:
    d_0 = 999  -> timestep,
    d_1 = 3    -> RGB-channel (= u_x, u_y, u_z),
    d_2 = 24   -> x-pos,
    d_3 = 24   -> y-pos,
    d_4 = 24   -> z-pos

    Args:
        my_images:
          Object of type numpy array containing the timeseries of multichannel
          volumetric data.
    """

    def __init__(self, my_images):
        self.samples = my_images[::20]
        self.sample_images = self.samples[:-1, :, 1:-1, 1:-1, 1:-1]
        self.sample_masks = self.samples[1:, :, 1:-1, 1:-1, 1:-1]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_images[idx])
        mask = torch.from_numpy(self.sample_masks[idx])
        return image, mask


if __name__ == "__main__":
    a = np.zeros((1000, 256))
    for i in range(1000):
        a[i] = a[i] + i
    b = MyMamicoDataset_RNN_analysis(a, seq_length=10)
    for i in range(10):
        image, mask = b[i]
        print(image[:, 0])
        print('next i')
