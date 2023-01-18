import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import AE, AE_u_i, AE_u_x, AE_u_y, AE_u_z
from torchmetrics import MeanSquaredLogError
from utils import get_AE_loaders, losses2file, dataset2csv
from plotting import compareLossVsValid, plot_flow_profile

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


def train_AE(loader, model, optimizer, criterion, scaler, model_identifier, current_epoch):
    """The train_AE function trains the model and computes the average loss on
    the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss = 0
    _counter = 0

    for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
        t, c, h, d, w = _data_0.shape
        _data_u_x = torch.reshape(
            _data_0[:, 0, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_y = torch.reshape(
            _data_0[:, 1, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_z = torch.reshape(
            _data_0[:, 2, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_x = torch.reshape(
            _targ_0[:, 0, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_y = torch.reshape(
            _targ_0[:, 1, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_z = torch.reshape(
            _targ_0[:, 2, :, :, :], (t, 1, h, d, w)).to(device=device)

        _data_1 = torch.cat((_data_u_y, _data_u_z, _data_u_x), 1).to(device)
        _data_2 = torch.cat((_data_u_z, _data_u_x, _data_u_y), 1).to(device)
        _targ_1 = torch.cat((_targ_u_y, _targ_u_z, _targ_u_x), 1).to(device)
        _targ_2 = torch.cat((_targ_u_z, _targ_u_x, _targ_u_y), 1).to(device)

        _data = torch.cat((_data_0.to(device), _data_1.to(
            device), _data_2.to(device)), 0).float().to(device)
        _targ = torch.cat((_targ_0.to(device), _targ_1.to(
            device), _targ_2.to(device)), 0).float().to(device)

        with torch.cuda.amp.autocast():
            _pred = model(_data).float().to(device)
            _loss = criterion(_pred, _targ)
            _epoch_loss += _loss.item()
            _counter += 1

        _loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def train_AE_u_i(loader, model_x, model_y, model_z,
                 optimizer_x, optimizer_y, optimizer_z,
                 model_identifier_x, model_identifier_y, model_identifier_z,
                 criterion, scaler, current_epoch):
    """The train_AE function trains the model and computes the average loss on
    the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss_x = 0
    _epoch_loss_y = 0
    _epoch_loss_z = 0
    _counter = 0

    for _batch_idx, (_data_0, _targ_0) in enumerate(loader):
        t, c, h, d, w = _data_0.shape
        _data_u_x = torch.reshape(
            _data_0[:, 0, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_y = torch.reshape(
            _data_0[:, 1, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_z = torch.reshape(
            _data_0[:, 2, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_x = torch.reshape(
            _targ_0[:, 0, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_y = torch.reshape(
            _targ_0[:, 1, :, :, :], (t, 1, h, d, w)).to(device=device)
        _targ_u_z = torch.reshape(
            _targ_0[:, 2, :, :, :], (t, 1, h, d, w)).to(device=device)

        _data_1 = torch.cat((_data_u_y, _data_u_z, _data_u_x), 1).to(device)
        _data_2 = torch.cat((_data_u_z, _data_u_x, _data_u_y), 1).to(device)
        _targ_1 = torch.cat((_targ_u_y, _targ_u_z, _targ_u_x), 1).to(device)
        _targ_2 = torch.cat((_targ_u_z, _targ_u_x, _targ_u_y), 1).to(device)

        _data = torch.cat((_data_0.to(device), _data_1.to(
            device), _data_2.to(device)), 0).float().to(device)
        _data = torch.add(_data, 0.2).float().to(device)
        _targ = torch.cat((_targ_0.to(device), _targ_1.to(
            device), _targ_2.to(device)), 0).float().to(device)

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data).float().to(device=device)
            _preds_y = model_y(_data).float().to(device=device)
            _preds_z = model_z(_data).float().to(device=device)

            _preds_x = torch.add(_preds_x, -0.2).float().to(device=device)
            _preds_y = torch.add(_preds_y, -0.2).float().to(device=device)
            _preds_z = torch.add(_preds_z, -0.2).float().to(device=device)

            _preds_x = model_x(_data).float().to(device=device)

            _targs_x = torch.reshape(
                _targ[:, 0, :, :, :].float(), (3*t, 1, h, d, w)).to(device=device)
            _targs_y = torch.reshape(
                _targ[:, 1, :, :, :].float(), (3*t, 1, h, d, w)).to(device=device)
            _targs_z = torch.reshape(
                _targ[:, 2, :, :, :].float(), (3*t, 1, h, d, w)).to(device=device)

            _loss_x = criterion(_preds_x, _targs_x)
            _loss_y = criterion(_preds_y, _targs_y)
            _loss_z = criterion(_preds_z, _targs_z)

            _epoch_loss_x += _loss_x.item()
            _epoch_loss_y += _loss_y.item()
            _epoch_loss_z += _loss_z.item()
            _counter += 1

        _loss_x.backward(retain_graph=True)
        _loss_y.backward(retain_graph=True)
        _loss_z.backward(retain_graph=True)
        optimizer_x.step()
        optimizer_y.step()
        optimizer_z.step()
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
        optimizer_z.zero_grad()

    _avg_loss_x = _epoch_loss_x/_counter
    _avg_loss_y = _epoch_loss_y/_counter
    _avg_loss_z = _epoch_loss_z/_counter
    return _avg_loss_x, _avg_loss_y, _avg_loss_z


def valid_AE(loader, model, criterion, model_identifier):
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
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.

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
            _pred = _predictions.float()
            _targ = _targets.float()
            _loss = criterion(_pred, _targ)
            # print('Current batch loss: ', loss.item())
            _epoch_loss += _loss.item()
            _counter += 1

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def valid_AE_u_i(loader, model_x, model_y, model_z,
                 optimizer_x, optimizer_y, optimizer_z,
                 model_identifier_x, model_identifier_y, model_identifier_z,
                 criterion, scaler, current_epoch):
    """The valid_AE function computes the average loss on a given dataset
    without updating/optimizing the learnable model parameters.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        optimizer:
          The optimization algorithm applied during training.
        criterion:
          The loss function applied to quantify the error.
        scaler:
          Object of torch.cuda.amp.GradScaler to conveniently help perform the
          steps of gradient scaling.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.
        current_epoch:
          A string containing the current epoch for terminal output.

    Returns:
        avg_loss:
          A double value indicating average training loss for the current epoch.
    """

    _epoch_loss_x = 0
    _epoch_loss_y = 0
    _epoch_loss_z = 0
    _counter = 0

    for _batch_idx, (_data, _targets) in enumerate(loader):
        _data = _data.float().to(device=device)
        _targets = _targets.float().to(device=device)
        t, c, h, d, w = _data.shape

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data).float().to(device=device)
            _preds_y = model_y(_data).float().to(device=device)
            _preds_z = model_z(_data).float().to(device=device)

            _targs_x = torch.reshape(
                _targets[:, 0, :, :, :].float(), (t, 1, h, d, w)).to(device=device)
            _targs_y = torch.reshape(
                _targets[:, 1, :, :, :].float(), (t, 1, h, d, w)).to(device=device)
            _targs_z = torch.reshape(
                _targets[:, 2, :, :, :].float(), (t, 1, h, d, w)).to(device=device)

            _loss_x = criterion(_preds_x, _targs_x)
            _loss_y = criterion(_preds_y, _targs_y)
            _loss_z = criterion(_preds_z, _targs_z)

            _epoch_loss_x += _loss_x.item()
            _epoch_loss_y += _loss_y.item()
            _epoch_loss_z += _loss_z.item()
            _counter += 1

    _avg_loss_x = _epoch_loss_x/_counter
    _avg_loss_y = _epoch_loss_y/_counter
    _avg_loss_z = _epoch_loss_z/_counter
    return _avg_loss_x, _avg_loss_y, _avg_loss_z


def error_timeline(loader, model, criterion):
    """The error_timeline function computes the individual errors on a dataset
    and returns a list containing all errors in chronological order.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch MOdule class, i.e. the model to be trained.
        criterion:
          The loss function applied to quantify the error.

    Returns:
        losses:
          A list containing all individual errors in chronological order.
    """

    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            losses.append(loss.item())

    return losses


def get_latentspace_AE(loader, model, out_file_name):
    """The get_latentspace_AE function extracts the model-specific latentspace
    for a given dataset and saves it to file.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch MOdule class, i.e. the model to be trained.
        out_file_name:
          A string containing the name of the file that the latentspace should
          be saved to.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          latentspace to file.
    """
    latentspace = []

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            bottleneck, _ = model(data,  y='get_bottleneck')
            latentspace.append(bottleneck.cpu().detach().numpy())

    np_latentspace = np.vstack(latentspace)
    dataset2csv(
        dataset=np_latentspace,
        dataset_name=f'{out_file_name}'
    )
    return


def get_latentspace_AE_helper():
    """The get_latentspace_AE_helper function contains the additional steps to
    create the model-specific latentspace. It loads an already trained model in
    model.eval() mode, loads the dataset loaders and calls the get_latentspace_AE
    function for each individual subdataset in the training and validation
    datasets.

    Args:
        NONE

    Returns:
        NONE:
    """
    print('Starting Trial 1: Get Latentspace (Couette)')
    _model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    _model.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/'
        '3_Constituent_Hybrid_approach/Results/1_AE/Model_AE_LR0_0005'))
    _model.eval()

    _loader_1, _loader_2_ = get_AE_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/Latentspace_Dataset'
    _out_file_names = [
        '_C_0_5_T',
        '_C_0_5_M',
        '_C_0_5_B',
        '_C_1_0_T',
        '_C_1_0_M',
        '_C_1_0_B',
        '_C_2_0_T',
        '_C_2_0_M',
        '_C_2_0_B',
        '_C_3_0_T',
        '_C_3_0_M',
        '_C_3_0_B',
        '_C_4_0_T',
        '_C_4_0_M',
        '_C_4_0_B',
        '_C_5_0_T',
        '_C_5_0_M',
        '_C_5_0_B',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE(
            loader=_loader,
            model=_model,
            out_file_name=f'{_out_directory}{_out_file_names[idx]}'
        )


def trial_1_AE(alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_1_AE function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

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
    _criterion = nn.L1Loss().to(device)
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/' + \
        '4_ICCS/Results/1_Conv_AE/'
    _model_identifier = f'LR{alpha_string}'
    print('Initializing AE model.')
    _model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.LeakyReLU(negative_slope=0.1, inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(4):
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
        file_name=f'{_file_prefix}Losses_AE_u_i_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_AE_u_i_{_model_identifier}'
    )

    compareLossVsValid(
        loss_files=[
            f'{_file_prefix}Losses_AE_u_i_{_model_identifier}.csv',
            f'{_file_prefix}Valids_AE_u_i_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'AE_u_i_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier}'
    )
    return


def trial_1_AE_u_i(alpha, alpha_string, train_loaders, valid_loaders):
    """The trial_1_AE function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

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
    _criterion = nn.L1Loss().to(device)
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/' + \
        '4_ICCS/Results/1_Conv_AE/'

    _model_identifier_x = f'LR{alpha_string}_x'
    _model_identifier_y = f'LR{alpha_string}_y'
    _model_identifier_z = f'LR{alpha_string}_z'

    print('Initializing AE_u_x/y/z model.')

    _model_x = AE_u_x(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_y = AE_u_y(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_z = AE_u_z(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer_x = optim.Adam(_model_x.parameters(), lr=alpha)
    _optimizer_y = optim.Adam(_model_y.parameters(), lr=alpha)
    _optimizer_z = optim.Adam(_model_z.parameters(), lr=alpha)

    print('Beginning training.')
    for epoch in range(10):
        _avg_loss_x = 0
        _avg_loss_y = 0
        _avg_loss_z = 0

        for _train_loader in train_loaders:
            _loss_x, _loss_y, _loss_z = train_AE_u_i(
                loader=_train_loader,
                model_x=_model_x,
                model_y=_model_y,
                model_z=_model_z,
                optimizer_x=_optimizer_x,
                optimizer_y=_optimizer_y,
                optimizer_z=_optimizer_z,
                model_identifier_x=_model_identifier_x,
                model_identifier_y=_model_identifier_y,
                model_identifier_z=_model_identifier_z,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _avg_loss_x += _loss_x
            _avg_loss_y += _loss_y
            _avg_loss_z += _loss_z

        _avg_loss_x = _avg_loss_x/len(train_loaders)
        _avg_loss_y = _avg_loss_y/len(train_loaders)
        _avg_loss_z = _avg_loss_z/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'Training Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')
        print(f'-> Avg u_y {_avg_loss_y:.3f}')
        print(f'-> Avg u_z {_avg_loss_z:.3f}')

        _sum_loss_x = 0
        _sum_loss_y = 0
        _sum_loss_z = 0
        for _valid_loader in valid_loaders:
            _loss_x, _loss_y, _loss_z = valid_AE_u_i(
                loader=_train_loader,
                model_x=_model_x,
                model_y=_model_y,
                model_z=_model_z,
                optimizer_x=_optimizer_x,
                optimizer_y=_optimizer_y,
                optimizer_z=_optimizer_z,
                model_identifier_x=_model_identifier_x,
                model_identifier_y=_model_identifier_y,
                model_identifier_z=_model_identifier_z,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _sum_loss_x += _loss_x
            _sum_loss_y += _loss_y
            _sum_loss_z += _loss_z

        _avg_valid_x = _sum_loss_x/len(train_loaders)
        _avg_valid_y = _sum_loss_y/len(train_loaders)
        _avg_valid_z = _sum_loss_z/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_valid_x:.3f}')
        print(f'-> Avg u_y {_avg_valid_y:.3f}')
        print(f'-> Avg u_z {_avg_valid_z:.3f}')

    '''
    losses2file(
        losses=_epoch_losses,
        file_name=f'{_file_prefix}Losses_AE_u_i_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_AE_u_i_{_model_identifier}'
    )

    compareLossVsValid(
        loss_files=[
            f'{_file_prefix}Losses_AE_u_i_{_model_identifier}.csv',
            f'{_file_prefix}Valids_AE_u_i_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'AE_u_i_{_model_identifier}'
    )
    '''
    torch.save(
        _model_x.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier_x}'
    )
    torch.save(
        _model_y.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier_y}'
    )
    torch.save(
        _model_z.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier_z}'
    )
    return


def trial_1_AE_mp():
    """The trial_1_AE_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_1_AE function. Here, 6 unique models are trained using
    the 6 learning rates (_alphas) respectively. Refer to the trial_1_AE
    function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 1: AE (Couette)')
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_couette',
        batch_size=32,
        shuffle=True
    )

    _processes = []

    for i in range(6):
        _p = mp.Process(
            target=trial_1_AE,
            args=(_alphas[i], _alpha_strings[i],
                  _train_loaders, _valid_loaders,)
        )
        _p.start()
        _processes.append(_p)
        print(f'Creating Process Number: {i+1}')

    for _process in _processes:
        _process.join()
        print('Joining Process')
    return


def trial_1_error_timeline():
    """The trial_1_error_timeline function is essentially a helper function to
    facilitate the error_timeline function with the desired model and datasets.
    Refer to the error_timeline function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 1: UNET AE Error Timeline')

    _directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/' + \
                 '4_ICCS/Results/1_Conv_AE/'
    _model_name = 'Model_AE_LR0_0005'
    _model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model.load_state_dict(torch.load(f'{_directory}{_model_name}'))
    _criterion = nn.L1Loss()
    _, _valid_loaders = get_AE_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )

    _datasets = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]

    for idx, _loader in enumerate(_valid_loaders):
        _losses = error_timeline(
            loader=_loader,
            model=_model,
            criterion=_criterion
        )
        losses2file(
            losses=_losses,
            file_name=f'{_directory}{_model_name}_Valid_Error_Timeline_{_datasets[idx]}'
        )

    pass


def prediction_retriever(model_directory, model_name, dataset_name, save2file_name):
    """The prediction_retriever function is used to evaluate model performance
    of a trained model. This is done by loading the saved model, feeding it
    with datasets and then saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name:

        dataset_name:

    Returns:
        NONE
    """
    _, valid_loaders = get_AE_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.LeakyReLU(negative_slope=0.1, inplace=True)
    ).to(device)
    _model.load_state_dict(torch.load(
        f'{model_directory}/{model_name}', map_location='cpu'))
    _model.eval()

    for i in range(len(valid_loaders)):
        _preds = []
        _targs = []
        for batch_idx, (data, target) in enumerate(valid_loaders[i]):
            data = data.float().to(device=device)
            target = target.float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred = _model(data)
                data_targ = target
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(data_targ.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)

    plot_flow_profile(_preds, save2file_name)


def prediction_retriever_u_i(model_directory, model_name_x, model_name_y,
                             model_name_z, dataset_name, save2file_name):
    """The prediction_retriever function is used to evaluate model performance
    of a trained model. This is done by loading the saved model, feeding it
    with datasets and then saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name:

        dataset_name:

    Returns:
        NONE
    """
    _, valid_loaders = get_AE_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _model_x = AE_u_x(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_y = AE_u_y(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_z = AE_u_z(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_x.load_state_dict(torch.load(
        f'{model_directory}/{model_name_x}', map_location='cpu'))
    _model_x.eval()
    _model_y.load_state_dict(torch.load(
        f'{model_directory}/{model_name_y}', map_location='cpu'))
    _model_y.eval()
    _model_z.load_state_dict(torch.load(
        f'{model_directory}/{model_name_z}', map_location='cpu'))
    _model_z.eval()

    for i in range(len(valid_loaders)):
        _preds = []
        for batch_idx, (data, target) in enumerate(valid_loaders[i]):
            data = data.float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data)
                data_pred_y = _model_y(data)
                data_pred_z = _model_z(data)
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                _preds.append(data_pred.cpu().detach().numpy())
        _preds = np.vstack(_preds)

    plot_flow_profile(_preds, save2file_name)


if __name__ == "__main__":
    '''
    print('Starting Trial 1: AE (KVS + Aug, MAE, LRLU)')

    _alpha = 0.0001
    _alpha_string = '0_0001'
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    trial_1_AE(_alpha, _alpha_string, _train_loaders, _valid_loaders)

    print('Starting Trial 1: Prediction Retriever (KVS + Aug, MAE, LReLU, AE)')

    _model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_04_mae_lrelu/'
    _model_name = 'Model_AE_u_i_LR0_0001'
    _dataset_name = 'get_KVS_eval'
    _save2file_name = 'pred_10_lrelu_kvs_aug_combined_domain_init_20000_NW'

    prediction_retriever(
        model_directory=_model_directory,
        model_name=_model_name,
        dataset_name=_dataset_name,
        save2file_name=_save2file_name
    )
    '''
    '''
    print('Starting Trial 1: AE_u_i (KVS + Aug, MAE, ReLU, torch.add())')
    _alpha = 0.0001
    _alpha_string = '0_0001'
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    trial_1_AE_u_i(_alpha, _alpha_string, _train_loaders, _valid_loaders)
    '''
    print('Starting Trial 1: Prediction Retriever (KVS + Aug, MAE, LReLU, AE_u_i, torch.add())')

    _model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_10_mae_relu_upshift/'
    _model_name_x = 'Model_AE_u_i_LR0_0001_x'
    _model_name_y = 'Model_AE_u_i_LR0_0001_y'
    _model_name_z = 'Model_AE_u_i_LR0_0001_z'
    _dataset_name = 'get_KVS_eval'
    _save2file_name = 'pred_10_relu_kvs_aug_upshift_combined_domain_init_20000_NW'

    prediction_retriever_u_i(
        model_directory=_model_directory,
        model_name_x=_model_name_x,
        model_name_y=_model_name_y,
        model_name_z=_model_name_z,
        dataset_name=_dataset_name,
        save2file_name=_save2file_name
    )
