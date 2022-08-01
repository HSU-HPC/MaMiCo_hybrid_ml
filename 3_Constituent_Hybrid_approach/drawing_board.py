import numpy as np
import torch
import torch.nn as nn
import glob
# , compareErrorTimeline
from plotting import compareAvgLoss, compareAvgLossRNN, compareLossVsValidRNN, compareFlowProfile3x3, compareLossVsValid
from model import UNET_AE, LSTM, Hybrid_MD_RNN_UNET
from utils_new import get_UNET_AE_loaders, get_Hybrid_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trial_1_UNET_AE_plots():
    model_names = [
        'Model_UNET_AE_LR0_005',
        'Model_UNET_AE_LR0_001',
        'Model_UNET_AE_LR0_0005',
        'Model_UNET_AE_LR0_0001',
    ]

    dataset_names = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]
    model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_UNET_AE/'

    loss_prefix = 'Losses_UNET_AE_LR'

    loss_files = [
        f'{model_directory}{loss_prefix}0_01.csv',
        f'{model_directory}{loss_prefix}0_001.csv',
        f'{model_directory}{loss_prefix}0_0001.csv',
        f'{model_directory}{loss_prefix}0_005.csv',
        f'{model_directory}{loss_prefix}0_0005.csv',
        f'{model_directory}{loss_prefix}0_00005.csv'
    ]
    loss_labels = [
        'Learning Rate = 0.01',
        'Learning Rate = 0.001',
        'Learning Rate = 0.0001',
        'Learning Rate = 0.005',
        'Learning Rate = 0.0005',
        'Learning Rate = 0.00005'
    ]
    '''
    compareAvgLoss(
        loss_files=loss_files,
        loss_labels=loss_labels,
        file_prefix=model_directory,
        file_name='UNET_AE'
    )
    '''
    lossVsValidFiles = [
        f'{model_directory}Losses_UNET_AE_LR0_001.csv',
        f'{model_directory}Valids_UNET_AE_LR0_001.csv',
        f'{model_directory}Losses_UNET_AE_LR0_0005.csv',
        f'{model_directory}Valids_UNET_AE_LR0_0005.csv'
    ]
    lossVsValidLabels = [
        'Training Loss   LR = 0.001',
        'Validation Loss LR = 0.001',
        'Training Loss   LR = 0.0005',
        'Validation Loss LR = 0.0005'
    ]

    compareLossVsValid(
        loss_files=lossVsValidFiles,
        loss_labels=lossVsValidLabels,
        file_prefix=model_directory,
        file_name='Model_UNET_AE_LR0_001_and_LR0_0005'
    )

    _, valid_loaders = get_UNET_AE_loaders(
            data_distribution='get_couette',
            batch_size=1,
            shuffle=False
        )
    for i in range(2, 3):

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

        for j in range(5, len(valid_loaders)):
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
            # _preds = np.array(
            #     [_preds[60], _preds[125], _preds[250], _preds[500], _preds[-1]])
            # _targs = np.array(
            #     [_targs[60], _targs[125], _targs[250], _targs[500], _targs[-1]])
            compareFlowProfile3x3(
                preds=_preds,
                targs=_targs,
                model_id=model_names[i],
                dataset_id=dataset_names[j],
            )


def trial_2_RNN_plots():
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_RNN/'

    _list_of_list_f = []
    _list_of_list_l = []
    '''
    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_RNN_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='RNN'
    )
    '''

    _list_of_list_f = []
    _list_of_list_l = []

    for _rnn_depth in _rnn_depths:
        files = []
        labels = []
        for _seq_length in _seq_lengths:
            files.append(
                f'{_directory}Losses_RNN_LR{_alpha_strings[-2]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
            labels.append(
                f'Training Seq:{_seq_length}')
            files.append(
                f'{_directory}Valids_RNN_LR{_alpha_strings[-2]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
            labels.append(
                f'Validation Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareLossVsValidRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name=f'And_Valids_RNN_LR{_alpha_strings[-2]}'
    )
    pass


def trial_3_GRU_plots():

    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/2_GRU/'

    _list_of_list_f = []
    _list_of_list_l = []
    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_GRU_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='GRU'
    )

    for _alpha_string in _alpha_strings:
        _list_of_list_f = []
        _list_of_list_l = []
        for _rnn_depth in _rnn_depths:
            files = []
            labels = []
            for _seq_length in _seq_lengths:
                # if _rnn_depth == 1 and _seq_length != 5:
                #     continue
                files.append(
                    f'{_directory}Losses_GRU_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Training Seq:{_seq_length}')
                files.append(
                    f'{_directory}Valids_GRU_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Validation Seq:{_seq_length}')
            _list_of_list_f.append(files)
            _list_of_list_l.append(labels)

        compareLossVsValidRNN(
            l_of_l_files=_list_of_list_f,
            l_of_l_labels=_list_of_list_l,
            file_prefix=_directory,
            file_name=f'And_Valids_GRU_LR{_alpha_string}'
        )
    pass


def trial_4_LSTM_plots():
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_LSTM/'

    _list_of_list_f = []
    _list_of_list_l = []
    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_LSTM_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='LSTM'
    )

    for _alpha_string in _alpha_strings:
        _list_of_list_f = []
        _list_of_list_l = []
        for _rnn_depth in _rnn_depths:
            files = []
            labels = []
            for _seq_length in _seq_lengths:
                # if _rnn_depth == 1 and _seq_length != 5:
                #     continue
                files.append(
                    f'{_directory}Losses_LSTM_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Training Seq:{_seq_length}')
                files.append(
                    f'{_directory}Valids_LSTM_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Validation Seq:{_seq_length}')
            _list_of_list_f.append(files)
            _list_of_list_l.append(labels)

        compareLossVsValidRNN(
            l_of_l_files=_list_of_list_f,
            l_of_l_labels=_list_of_list_l,
            file_prefix=_directory,
            file_name=f'And_Valids_LSTM_LR{_alpha_string}'
        )
    pass


def trial_5_Hybrid_plots():
    _, valid_loaders = get_Hybrid_loaders(file_names=-1)
    # _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=0)
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/4_Hybrid_RNN_UNET/'
    _model_identifier = 'LSTM_Seq15_Lay1_LR0_00005'
    print('Initializing model.')

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/Model_UNET_AE_0_001'))

    _model_rnn = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=15,
        num_layers=1,
        device=device
    )
    _model_rnn.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_LSTM/Model_LSTM_Seq15_Lay1_LR0_00005'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=15
    ).to(device)

    _model_hybrid.eval()

    dataset_names = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]

    for j in range(len(valid_loaders)):
        _preds = []
        _targs = []
        for batch_idx, (data, targets) in enumerate(valid_loaders[j]):
            data = data.float().to(device=device)
            targets = targets.float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred = _model_hybrid(data)
                data_targ = targets
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(data_targ.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        # print('Shape of preds: ', preds.shape)
        _targs = np.vstack(_targs)
        # print('Shape of targs: ', targs.shape)
        # _preds = np.array(
        #     [_preds[60], _preds[125], _preds[250], _preds[500], _preds[-1]])
        # _targs = np.array(
        #     [_targs[60], _targs[125], _targs[250], _targs[500], _targs[-1]])

    pass


def trial_6_Hybrid_kvs_plots():
    _model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/6_Hybrid_KVS/'
    _lossVsValidFiles = [
        f'{_model_directory}Losses_KVS_RNN_LR0_00001_Lay1_Seq25.csv',
        f'{_model_directory}Valids_KVS_RNN_LR0_00001_Lay1_Seq25.csv',
        f'{_model_directory}Losses_KVS_GRU_LR0_00001_Lay2_Seq25.csv',
        f'{_model_directory}Valids_KVS_GRU_LR0_00001_Lay2_Seq25.csv',
        f'{_model_directory}Losses_KVS_LSTM_LR0_00001_Lay2_Seq25.csv',
        f'{_model_directory}Valids_KVS_LSTM_LR0_00001_Lay2_Seq25.csv',
    ]
    _lossVsValidLabels = [
        'Train. Loss   RNN',
        'Valid. Loss   RNN',
        'Train. Loss   GRU',
        'Valid. Loss   GRU',
        'Train. Loss   LSTM',
        'Valid. Loss   LSTM'
    ]

    compareLossVsValid(
        loss_files=_lossVsValidFiles,
        loss_labels=_lossVsValidLabels,
        file_prefix=_model_directory,
        file_name='Model_RNNs'
    )

    '''
    _alphas = [0.00001]
    _alpha_strings = ['0_00001']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/'

    _list_of_list_f = []
    _list_of_list_l = []
    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_LSTM_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='LSTM-Hybrid'
    )

    for _alpha_string in _alpha_strings:
        _list_of_list_f = []
        _list_of_list_l = []
        for _rnn_depth in _rnn_depths:
            files = []
            labels = []
            for _seq_length in _seq_lengths:
                # if _rnn_depth == 1 and _seq_length != 5:
                #     continue
                files.append(
                    f'{_directory}Losses_LSTM_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Training Seq:{_seq_length}')
                files.append(
                    f'{_directory}Valids_LSTM_LR{_alpha_string}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Validation Seq:{_seq_length}')
            _list_of_list_f.append(files)
            _list_of_list_l.append(labels)

        compareLossVsValidRNN(
            l_of_l_files=_list_of_list_f,
            l_of_l_labels=_list_of_list_l,
            file_prefix=_directory,
            file_name=f'And_Valids_LSTM_LR{_alpha_string}'
        )
    '''

    pass


def trial_7_Hybrid_KVS_non_UNET_plots():
    print('Trial 7: Hybrid KVS non UNET (plotting)')
    _model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/7_Hybrid_KVS_non_UNET/'
    _lossVsValidFiles = [
        f'{_model_directory}Losses_AE_Both_LR0_0005.csv',
        f'{_model_directory}Valids_AE_Both_LR0_0005.csv',
        f'{_model_directory}Losses_AE_Both_LR0_0001.csv',
        f'{_model_directory}Valids_AE_Both_LR0_0001.csv',
        f'{_model_directory}Losses_AE_Both_LR0_00005.csv',
        f'{_model_directory}Valids_AE_Both_LR0_00005.csv',
        f'{_model_directory}Losses_AE_Both_LR0_00001.csv',
        f'{_model_directory}Valids_AE_Both_LR0_00001.csv',
        f'{_model_directory}Losses_AE_Both_LR0_000005.csv',
        f'{_model_directory}Valids_AE_Both_LR0_000005.csv',
        f'{_model_directory}Losses_AE_Both_LR0_000001.csv',
        f'{_model_directory}Valids_AE_Both_LR0_000001.csv',
    ]
    _lossVsValidLabels = [
        'Train. Loss LR 0.0005',
        'Valid. Loss LR 0.0005',
        'Train. Loss LR 0.0001',
        'Valid. Loss LR 0.0001',
        'Train. Loss LR 0.00005',
        'Valid. Loss LR 0.00005',
        'Train. Loss LR 0.00001',
        'Valid. Loss LR 0.00001',
        'Train. Loss LR 0.000005',
        'Valid. Loss LR 0.000005',
        'Train. Loss LR 0.000001',
        'Valid. Loss LR 0.000001'
    ]

    compareLossVsValid(
        loss_files=_lossVsValidFiles,
        loss_labels=_lossVsValidLabels,
        file_prefix=_model_directory,
        file_name='Model_AE_non_UNET'
    )

    pass


def analysis_2_plots():
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/8_Analysis_2_Larger_Time_Intervals/'

    _list_of_list_f = []
    _list_of_list_l = []

    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_RNN_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='RNN'
    )

    _list_of_list_f = []
    _list_of_list_l = []

    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_GRU_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='GRU'
    )

    _list_of_list_f = []
    _list_of_list_l = []

    for idx, _alpha in enumerate(_alphas):
        files = []
        labels = []
        for _rnn_depth in _rnn_depths:
            for _seq_length in _seq_lengths:
                files.append(
                    f'{_directory}Losses_LSTM_LR{_alpha_strings[idx]}_Lay{_rnn_depth}_Seq{_seq_length}.csv')
                labels.append(
                    f'Lay:{_rnn_depth} Seq:{_seq_length}')
        _list_of_list_f.append(files)
        _list_of_list_l.append(labels)

    compareAvgLossRNN(
        l_of_l_files=_list_of_list_f,
        l_of_l_labels=_list_of_list_l,
        file_prefix=_directory,
        file_name='LSTM'
    )
    pass


def analysis_3_plots():
    model_names = [
        'Model_AE_LR0_0005',
        'Model_AE_LR0_0001',
        'Model_AE_LR0_00005',
        'Model_AE_LR0_00001',
        'Model_AE_LR0_000001',
    ]

    dataset_names = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]
    model_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/AE/'
    loss_prefix = 'Losses_AE_LR'

    loss_files = [
        f'{model_directory}{loss_prefix}0_0005.csv',
        f'{model_directory}{loss_prefix}0_0001.csv',
        f'{model_directory}{loss_prefix}0_00005.csv',
        f'{model_directory}{loss_prefix}0_00001.csv',
        f'{model_directory}{loss_prefix}0_000001.csv',
    ]
    loss_labels = [
        'Learning Rate = 0.0005',
        'Learning Rate = 0.0001',
        'Learning Rate = 0.00005',
        'Learning Rate = 0.00001',
        'Learning Rate = 0.000001',
    ]

    compareAvgLoss(
        loss_files=loss_files,
        loss_labels=loss_labels,
        file_prefix=model_directory,
        file_name='NON_UNET_AE'
    )


if __name__ == "__main__":
    '''_directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                 '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/RNNs/'
    _models = ['RNN', 'GRU', 'LSTM']
    _items = ['Losses', 'Valids']

    for _model in _models:
        for _item in _items:
            _csv_files = glob.glob(
                f"{_directory}{_item}_{_model}_LR0_*")
            _losses = []
            for _file in _csv_files:
                with open(_file, 'r') as f:
                    last_line = f.readlines()[-1]
                    # print(_file)
                    # print(last_line)
                    _losses.append(float(last_line))

            _min_loss = 5
            for _loss in _losses:
                _loss = float(_loss)
                if _loss < _min_loss and _loss > 0:
                    _min_loss = _loss
            _min_indx = _losses.index(_min_loss)
            _min_name = _csv_files[_min_indx].replace(_directory, '')
            print(f'Model: {_min_name}')
            print(f'Min Loss: {_min_loss}')
            print('------------------------------------------------------------')
    '''
    trial_1_UNET_AE_plots()
