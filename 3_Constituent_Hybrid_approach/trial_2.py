import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import RNN
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
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


def train_RNN(loader, model, optimizer, criterion, scaler, model_identifier, current_epoch):
    """The train_AE function trains the model and computes the average loss on
    the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch MOdule class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, a combination of the
          learning rate (_alpha), num of RNN layers (_num_layers) and sequence
          length (_seq_length) is used.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss = 0
    _counter = 0

    for _batch_idx, (_data, _targets) in enumerate(loader):
        _data = _data.float().to(device=device)
        _targets = _targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            _predictions = model(_data)
            _loss = criterion(_predictions.float(), _targets.float())
            _epoch_loss += _loss.item()
            _counter += 1

        _loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def valid_RNN(loader, model, criterion, model_identifier):
    """The valid_AE function computes the average loss on a given dataset
    without updating/optimizing the learnable model parameters.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch MOdule class, i.e. the model to be trained.
        criterion:
          The loss function applied to quantify the error.
        model_identifier:
          A unique string to identify the model. Here, a combination of the
          learning rate (_alpha), num of RNN layers (_num_layers) and sequence
          length (_seq_length) is used.

    Returns:
        avg_loss:
          A double value indicating average validation loss for the current epoch.
    """

    _epoch_loss = 0
    _counter = 0

    for _batch_idx, (_data, _targets) in enumerate(loader):
        _data = _data.float().to(device=device)
        _targets = _targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            _predictions = model(_data)
            _loss = criterion(_predictions.float(), _targets.float())
            # print('Current batch loss: ', loss.item())
            _epoch_loss += _loss.item()
            _counter += 1

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def trial_2_RNN(seq_length, num_layers, alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_2_RNN function trains the given model and documents its
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
                   '3_Constituent_Hybrid_approach/Results/2_RNN/'
    _model_identifier = f'LR{alpha_string}_Lay{num_layers}_Seq{seq_length}'
    print('Initializing RNN model.')
    _model = RNN(
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
        filename=f'{_file_prefix}Losses_RNN_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_RNN_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_RNN_{_model_identifier}.csv',
            f'{_file_prefix}Valids_RNN_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_RNN_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_RNN_{_model_identifier}'
    )


def trial_2_RNN_mp():
    """The trial_2_RNN_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_2_RNN function. Here, 54 unique models are trained using all
    possible combinations from the list of learning rates (_alphas), number of
    RNN layers (_rnn_Depths) and RNN sequence lengths (_seq_lengths). Refer to
    the trial_2_RNN function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 2: RNN')
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=0, sequence_length=5)
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx, _lr in enumerate(_alphas):
        _counter = 1

        for _rnn_depth in _rnn_depths:
            _processes = []

            for i in range(3):
                _p = mp.Process(
                    target=trial_2_RNN,
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
    trial_2_RNN_mp()
