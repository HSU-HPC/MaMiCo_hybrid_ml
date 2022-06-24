import numpy as np
import csv
import torch
import torch.nn as nn
from plotting import colorMap
from model import UNET_AE
from utils import get_UNET_AE_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trial_1_plots():
    _, valid_loaders = get_UNET_AE_loaders(file_names=-1)
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
                with torch.cuda.amp.autocast():
                    preds.append(_model(data).cpu().detach().numpy())
                    targs.append(targets.cpu().detach().numpy())
            preds = np.vstack(preds)
            print('Shape of preds: ', preds.shape)
            targs = np.vstack(targs)
            print('Shape of targs: ', targs.shape)

    pass


if __name__ == "__main__":
    trial_1_plots()
    pass
