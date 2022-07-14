import torch
import random
import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_UNET, resetPipeline
from utils_new import get_UNET_AE_loaders, get_RNN_loaders, losses2file, get_Hybrid_loaders
from plotting import compareAvgLoss, compareLossVsValid
from trial_1 import train_AE, valid_AE, error_timeline, get_latentspace_AE
from trial_2 import train_RNN, valid_RNN

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1


def trial_6_KVS_AE(alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_6_KVS_AE function trains the given model on the KVS data
    distribution and documents its progress via saving average training and
    validation losses to file and comparing them in a plot.

    Args:
        alpha:
          A double value indicating the chosen learning rate.
        alpha_string:
          Object of type string used as a model identifier.
        train_loaders:
          Object of PyTorch-type DataLoader to automatically pass training
          dataset to model.
        valid_loaders:
          Object of PyTorch-type DataLoader to automatically pass validation
          dataset to model.

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/6_Hybrid_KVS/'
    _model_identifier = f'LR{alpha_string}'
    print('Initializing UNET_AE model.')
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(50):
        _avg_loss = 0
        for _train_loader in train_loaders:
            _avg_loss += train_AE(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                model_identifier=_model_identifier,
                current_epoch=epoch+1
            )
        _avg_loss = _avg_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Training Epoch: {epoch+1} -> Averaged'
              f'Loader Loss: {_avg_loss:.3f}')
        _epoch_losses.append(_avg_loss)

        _avg_valid = 0
        for _valid_loader in valid_loaders:
            _avg_valid += valid_AE(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                model_identifier=_model_identifier
            )
        _avg_valid = _avg_valid/len(valid_loaders)
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss:'
              f'{_avg_valid:.3f}')
        _epoch_valids.append(_avg_valid)

    losses2file(
        losses=_epoch_losses,
        file_name=f'{_file_prefix}Losses_UNET_AE_KVS_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_UNET_AE_KVS_{_model_identifier}'
    )

    compareLossVsValid(
        loss_files=[
            f'{_file_prefix}Losses_UNET_AE_KVS_{_model_identifier}.csv',
            f'{_file_prefix}Valids_UNET_AE_KVS_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'UNET_AE_KVS_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_UNET_AE_KVS_{_model_identifier}'
    )
    return


def trial_6_KVS_AE_helper():
    """The trial_6_KVS_AE_helper function is essentially a helper function to
    facilitate the training of the most promising UNET_AE model configurations
    from trial_1 on the basis of the KVS data distribution.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 6: UNET AE (KVS)')
    _t_loaders, _v_loaders = get_UNET_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )
    trial_6_KVS_AE(
        alpha=0.0005,
        alpha_string='0_0005',
        train_loaders=_t_loaders,
        valid_loaders=_v_loaders
    )


def trial_6_KVS_AE_latentspace_helper():
    """The trial_6_KVS_AE_latentspace_helper function contains the additional
    steps to create the model-specific latentspace. It loads an already trained
    model in model.eval() mode, loads the dataset loaders and calls the
    get_latentspace_AE function for each individual subdataset in the training
    and validation datasets.

    Args:
        NONE

    Returns:
        NONE:
    """
    print('Starting Trial 6: Get Latentspace (KVS)')
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/6_Hybrid_KVS/'
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    _model.load_state_dict(torch.load(f'{_file_prefix}Model_UNET_AE_LR0_0005'))
    _model.eval()

    _loader_1, _loader_2_ = get_UNET_AE_loaders(
        file_names=-2,
        num_workers=1,
        batch_size=32,
        shuffle=False
    )
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/CleanKVSLS/Latentspace_Dataset'
    _out_file_names = [
        '_kvs_10K_NE',
        '_kvs_10K_NW',
        '_kvs_10K_SE',
        '_kvs_10K_SW',
        '_kvs_20K_NE',
        '_kvs_20K_NW',
        '_kvs_20K_SE',
        '_kvs_20K_SW',
        '_kvs_30K_NE',
        '_kvs_30K_NW',
        '_kvs_30K_SE',
        '_kvs_30K_SW',
        '_kvs_40K_NE',
        '_kvs_40K_NW',
        '_kvs_40K_SE',
        '_kvs_40K_SW',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE(
            loader=_loader,
            model=_model,
            out_file_name=f'{_out_directory}{_out_file_names[idx]}'
        )


def trial_6_KVS_error_timeline():
    """The trial_6_KVS_error_timeline function creates an error timeline for
    each validation dataset. The resulting error timelines are plotted via the
    compareErrorTimeline_np function.

    Args:
        NONE

    Returns:
        NONE
    """
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/6_Hybrid_KVS/'
    model_name_1 = 'Model_UNET_AE_LR0_0005'
    model_name_2 = 'Model_LSTM_LR0_00001_Lay3_Seq25'
    model_name_3 = 'Hybrid_Model'

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_unet.load_state_dict(torch.load(f'{_directory}{model_name_1}'))

    _model_rnn = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=3,
        device=device
    )
    _model_rnn.load_state_dict(torch.load(f'{_directory}{model_name_2}'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=25
    ).to(device)

    _criterion = nn.L1Loss()
    _, valid_loaders = get_Hybrid_loaders(file_names=-2)

    _datasets = [
        'kvs_40K_NE',
        'kvs_40K_NW',
        'kvs_40K_SE',
        'kvs_40K_SW'
    ]

    for idx, _loader in enumerate(valid_loaders):
        _losses = error_timeline(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion
        )
        losses2file(
            losses=_losses,
            file_name=f'{_directory}{model_name_3}_KVS_Valid_Error_Timeline_{_datasets[idx]}'
        )

    pass


def trial_6_KVS_RNN(model, model_identifier, alpha, train_loaders, valid_loaders):
    """The trial_6_KVS_RNN function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        model_identifier:
          A unique string to identify the model. Here, a combination of the
          learning rate (_alpha), num of RNN layers (_num_layers) and sequence
          length (_seq_length) is used.
        alpha:
          A double value indicating the chosen learning rate.
        train_loaders:
          Object of PyTorch-type DataLoader to automatically pass training
          dataset to model.
        valid_loaders:
          Object of PyTorch-type DataLoader to automatically pass validation
          dataset to model.

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/6_Hybrid_KVS/'

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(model.parameters(), lr=alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(50):
        _avg_loss = 0
        for _train_loader in train_loaders:
            _avg_loss += train_RNN(
                loader=_train_loader,
                model=model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                model_identifier=model_identifier,
                current_epoch=epoch+1
            )
        _avg_loss = _avg_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(
            f'{model_identifier} Training Epoch: {epoch+1}-> Averaged '
            f'Loader Loss: {_avg_loss:.3f}')

        _epoch_losses.append(_avg_loss)

        _avg_valid = 0
        for _valid_loader in valid_loaders:
            _avg_valid += valid_RNN(
                loader=_valid_loader,
                model=model,
                criterion=_criterion,
                model_identifier=model_identifier
            )
        _avg_valid = _avg_valid/len(valid_loaders)
        print('------------------------------------------------------------')
        print(f'{model_identifier} Validation -> Averaged '
              f'Loader Loss: {_avg_valid:.3f}')
        _epoch_valids.append(_avg_valid)

    losses2file(
        losses=_epoch_losses,
        file_name=f'{_file_prefix}Losses_{model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_{model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_{model_identifier}.csv',
            f'{_file_prefix}Valids_{model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_{model_identifier}'
    )
    torch.save(
        model.state_dict(),
        f'{_file_prefix}Model_{model_identifier}'
    )


def trial_6_KVS_RNN_mp():
    """The trial_6_KVS_RNN_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_6_KVS_RNN function. Here, 3 unique models are trained using
    the most promising RNN/GRU/LSTM configurations from trials 2/3/4. Refer to
    the trial_6_KVS_RNN function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    _models = []
    _model_identifiers = [
        'KVS_RNN_LR0_00001_Lay1_Seq25',
        'KVS_GRU_LR0_00001_Lay2_Seq25',
        'KVS_LSTM_LR0_00001_Lay2_Seq25',
    ]
    _alphas = [0.00001, 0.00001, 0.00001]

    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=1,
        device=device
    ).to(device)
    _models.append(_model_rnn_1)
    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    ).to(device)
    _models.append(_model_rnn_2)
    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    ).to(device)
    _models.append(_model_rnn_3)

    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        seq_length=25
    )
    processes = []

    for i in range(3):
        p = mp.Process(
            target=trial_6_KVS_RNN,
            args=(_models[i], _model_identifiers[i], _alphas[i],
                  _t_loader_25, _v_loader_25,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {i+1}')

    for process in processes:
        process.join()
        print('Joining Process')


if __name__ == "__main__":

    trial_6_KVS_AE_helper()
