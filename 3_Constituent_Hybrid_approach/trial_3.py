import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import GRU
from trial_2 import train_RNN, valid_RNN
from utils import get_RNN_loaders, losses2file
from plotting import compareAvgLoss

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def trial_3_GRU(seq_length, num_layers, alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_3_GRU function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        seq_length:
          Object of integer type specifying the number of elements to include
          in the RNN sequence.
        num_layers:
          Object of integer type specifying the number of RNN layers to include
          in the RNN model.
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
                   '3_Constituent_Hybrid_approach/Results/3_GRU/'
    _model_identifier = f'LR{alpha_string}_Lay{num_layers}_Seq{seq_length}'
    print('Initializing GRU model.')
    _model = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=seq_length,
        num_layers=num_layers,
        device=device
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
            _avg_loss += train_RNN(
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
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged '
            f'Loader Loss: {_avg_loss:.3f}')

        _epoch_losses.append(_avg_loss)

        _avg_valid = 0
        for _valid_loader in valid_loaders:
            _avg_valid += valid_RNN(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                model_identifier=_model_identifier
            )
        _avg_valid = _avg_valid/len(valid_loaders)
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Validation -> Averaged Loader Loss: {_avg_valid:.3f}')
        _epoch_valids.append(_avg_valid)

    losses2file(
        losses=_epoch_losses,
        file_name=f'{_file_prefix}Losses_GRU_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_GRU_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_GRU_{_model_identifier}.csv',
            f'{_file_prefix}Valids_GRU_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_GRU_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_GRU_{_model_identifier}'
    )


def trial_3_GRU_mp():
    """The trial_3_GRU_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_3_GRU function. Here, 54 unique models are trained using all
    possible combinations from the list of learning rates (_alphas), number of
    RNN layers (_rnn_Depths) and RNN sequence lengths (_seq_lengths). Refer to
    the trial_3_GRU function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 3: GRU')
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=5
    )
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=15
    )
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=25
    )

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx, _lr in enumerate(_alphas):
        _counter = 1

        for _rnn_depth in _rnn_depths:
            _processes = []

            for i in range(3):
                _p = mp.Process(
                    target=trial_3_GRU,
                    args=(_seq_lengths[i], _rnn_depth, _lr,
                          _alpha_strings[idx], _t_loaders[i], _v_loaders[i],)
                )
                _p.start()
                _processes.append(_p)
                print(f'Creating Process Number: {_counter}')
                _counter += 1

        for _process in _processes:
            _process.join()
            print('Joining Process')


if __name__ == "__main__":
    trial_3_GRU_mp()
