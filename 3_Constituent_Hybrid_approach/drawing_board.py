import numpy as np
import csv
import torch
import torch.nn as nn
from plotting import colorMap
from model import UNET_AE
from utils import get_mamico_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trial_1_plots():
    _, valid_loaders = get_mamico_loaders(file_names=-1)
    model_names = [
        'Model_UNET_AE_0_01',
        'Model_UNET_AE_0_005',
        'Model_UNET_AE_0_001',
        'Model_UNET_AE_0_0005',
        'Model_UNET_AE_0_0001',
        'Model_UNET_AE_0_00005',
    ]
    model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'

    for model_name in model_names:

        _model = UNET_AE(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        )
        _model.load_state_dict(torch.load(
            f'{model_directory}{model_name}'))
        _model.eval()

        for loader in valid_loaders:
            preds = []
            targs = []
            for batch_idx, (data, targets) in enumerate(loader):
                preds.append(_model(data).cpu().detach().numpy())
                targs.append(targets.cpu().detach().numpy())
            preds = np.vstack(preds)
            print('Shape of preds: ', preds.shape)
            targs = np.vstack(targs)
            print('Shape of targs: ', targs.shape)

    pass


if __name__ == "__main__":
    array_a = np.random.rand(500, 3, 26, 26, 26)
    array_b = np.stack((array_a[0], array_a[249], array_a[-1]))
    print('Shape of array_b: ', array_b.shape)
    if np.array_equal(array_a[0], array_b[0]):
        print('First element equal.')
    if np.array_equal(array_a[249], array_b[1]):
        print('Second element equal.')
    if np.array_equal(array_a[-1], array_b[-1]):
        print('Third element equal.')
    pass
