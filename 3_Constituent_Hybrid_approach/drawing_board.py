import numpy as np
import csv
import torch
import torch.nn as nn
from plotting import compareColorMap, compareAvgLoss, compareAvgLossRNN
from model import UNET_AE
from utils import get_UNET_AE_loaders, csv2dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trial_0_UNET_AE_plots():
    _, valid_loaders = get_UNET_AE_loaders(file_names=-1)
    model_names = [
        'Model_UNET_AE_0_01',
        'Model_UNET_AE_0_005',
        'Model_UNET_AE_0_001',
        'Model_UNET_AE_0_0005',
        'Model_UNET_AE_0_0001',
        'Model_UNET_AE_0_00005'
    ]
    dataset_names = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]
    model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'

    for i in range(len(model_names)):

        _model = UNET_AE(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        ).to(device)
        _model.load_state_dict(torch.load(
            f'{model_directory}{model_names[i]}'))
        _model.eval()

        for j in range(len(valid_loaders)):
            _preds = []
            _targs = []
            for batch_idx, (data, targets) in enumerate(valid_loaders[j]):
                data = data.float().to(device=device)
                targets = targets.float().to(device=device)
                with torch.cuda.amp.autocast():
                    data_pred = _model(data)
                    data_targ = targets
                    _preds.append(data_pred.cpu().detach().numpy())
                    _targs.append(data_targ.cpu().detach().numpy())
            _preds = np.vstack(_preds)
            # print('Shape of preds: ', preds.shape)
            _targs = np.vstack(_targs)
            # print('Shape of targs: ', targs.shape)
            _preds = np.array(
                [_preds[60], _preds[125], _preds[250], _preds[500], _preds[-1]])
            _targs = np.array(
                [_targs[60], _targs[125], _targs[250], _targs[500], _targs[-1]])
            compareColorMap(
                preds=_preds,
                targs=_targs,
                model_name=model_names[i],
                dataset_name=dataset_names[j]
            )
    pass


def trial_1_RNN_plots():
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _rnn_depths = [1, 2, 3, 4]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_RNN/'

    _list_of_list_f = []
    _list_of_list_l = []
    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_RNN_Seq{_seq_length}_Lay{_rnn_depth}_LR{_alpha_strings[idx]}.csv')
                labels.append(
                    f'Seq:{_seq_length} Lay:{_rnn_depth} LR:{_alpha}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        loss_files=files,
        loss_labels=labels,
        file_prefix=_directory,
        file_name=f'LR{_alpha_strings[idx]}'
    )

    pass


def trial_2_GRU_plots():
    pass


def trial_3_LSTM_plots():
    pass


if __name__ == "__main__":
    trial_1_RNN_plots()
    pass
