import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import AE, AE_u_i, AE_u_y, AE_u_z
from torchmetrics import MeanSquaredLogError
from utils import get_AE_loaders, losses2file, dataset2csv
from plotting import compareLossVsValid, plot_flow_profile, plotPredVsTargKVS

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


def train_AE_u_i(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The train_AE function trains the single channel model and computes the
    average loss on the training set.

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

        _data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _data = torch.add(_data, 1.0).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = torch.reshape(_targ, (t*c, 1, h, d, w)).float().to(device)

        with torch.cuda.amp.autocast():
            _pred = model_i(_data).float().to(device=device)
            _pred = torch.add(_pred, -1.0).float().to(device=device)

            _loss = criterion(_pred, _targ)

            _epoch_loss += _loss.item()
            _counter += 1

        _loss.backward(retain_graph=True)
        optimizer_i.step()
        optimizer_i.zero_grad()

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


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


def valid_AE_u_i(loader, model_i, optimizer_i, model_identifier_i, criterion, scaler, current_epoch):
    """The valid_AE_u_i function computes the average loss on a given single
    channel dataset without updating/optimizing the learnable model parameters.

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

        _data = _data_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _data = torch.add(_data, 1.0).float().to(device)
        _targ = _targ_0.flatten(start_dim=0, end_dim=1).float().to(device)
        _targ = torch.reshape(_targ, (t*c, 1, h, d, w)).float().to(device)

        with torch.cuda.amp.autocast():
            _pred = model_i(_data).float().to(device=device)
            _pred = torch.add(_pred, -1.0).float().to(device=device)

            _loss = criterion(_pred, _targ)

            _epoch_loss += _loss.item()
            _counter += 1

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


def get_latentspace_AE_u_i(loader, model_x, model_y, model_z, out_file_name):
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
    latentspace_x_0 = []
    latentspace_x_1 = []
    latentspace_x_2 = []
    latentspace_y_0 = []
    latentspace_y_1 = []
    latentspace_y_2 = []
    latentspace_z_0 = []
    latentspace_z_1 = []
    latentspace_z_2 = []

    for batch_idx, (_data_0, _) in enumerate(loader):
        t, c, h, d, w = _data_0.shape
        _data_u_x = torch.reshape(
            _data_0[:, 0, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_y = torch.reshape(
            _data_0[:, 1, :, :, :], (t, 1, h, d, w)).to(device=device)
        _data_u_z = torch.reshape(
            _data_0[:, 2, :, :, :], (t, 1, h, d, w)).to(device=device)

        _data_0 = _data_0.float().to(device)
        _data_1 = torch.cat(
            (_data_u_y, _data_u_z, _data_u_x), 1).float().to(device)
        _data_2 = torch.cat(
            (_data_u_z, _data_u_x, _data_u_y), 1).float().to(device)

        with torch.cuda.amp.autocast():
            bottleneck_x_0 = model_x(_data_0,  y='get_bottleneck')
            bottleneck_x_1 = model_x(_data_1,  y='get_bottleneck')
            bottleneck_x_2 = model_x(_data_2,  y='get_bottleneck')
            latentspace_x_0.append(bottleneck_x_0.cpu().detach().numpy())
            latentspace_x_1.append(bottleneck_x_1.cpu().detach().numpy())
            latentspace_x_2.append(bottleneck_x_2.cpu().detach().numpy())

            bottleneck_y_0 = model_y(_data_0,  y='get_bottleneck')
            bottleneck_y_1 = model_y(_data_1,  y='get_bottleneck')
            bottleneck_y_2 = model_y(_data_2,  y='get_bottleneck')
            latentspace_y_0.append(bottleneck_y_0.cpu().detach().numpy())
            latentspace_y_1.append(bottleneck_y_1.cpu().detach().numpy())
            latentspace_y_2.append(bottleneck_y_2.cpu().detach().numpy())

            bottleneck_z_0 = model_z(_data_0,  y='get_bottleneck')
            bottleneck_z_1 = model_z(_data_1,  y='get_bottleneck')
            bottleneck_z_2 = model_z(_data_2,  y='get_bottleneck')
            latentspace_z_0.append(bottleneck_z_0.cpu().detach().numpy())
            latentspace_z_1.append(bottleneck_z_1.cpu().detach().numpy())
            latentspace_z_2.append(bottleneck_z_2.cpu().detach().numpy())

    np_latentspace_x_0 = np.vstack(latentspace_x_0)
    np_latentspace_x_1 = np.vstack(latentspace_x_1)
    np_latentspace_x_2 = np.vstack(latentspace_x_2)

    dataset2csv(
        dataset=np_latentspace_x_0,
        dataset_name=f'{out_file_name}_x_0'
    )
    dataset2csv(
        dataset=np_latentspace_x_1,
        dataset_name=f'{out_file_name}_x_1'
    )
    dataset2csv(
        dataset=np_latentspace_x_2,
        dataset_name=f'{out_file_name}_x_2'
    )

    np_latentspace_y_0 = np.vstack(latentspace_y_0)
    np_latentspace_y_1 = np.vstack(latentspace_y_1)
    np_latentspace_y_2 = np.vstack(latentspace_y_2)

    dataset2csv(
        dataset=np_latentspace_y_0,
        dataset_name=f'{out_file_name}_y_0'
    )
    dataset2csv(
        dataset=np_latentspace_y_1,
        dataset_name=f'{out_file_name}_y_1'
    )
    dataset2csv(
        dataset=np_latentspace_y_2,
        dataset_name=f'{out_file_name}_y_2'
    )

    np_latentspace_z_0 = np.vstack(latentspace_z_0)
    np_latentspace_z_1 = np.vstack(latentspace_z_1)
    np_latentspace_z_2 = np.vstack(latentspace_z_2)

    dataset2csv(
        dataset=np_latentspace_z_0,
        dataset_name=f'{out_file_name}_z_0'
    )
    dataset2csv(
        dataset=np_latentspace_z_1,
        dataset_name=f'{out_file_name}_z_1'
    )
    dataset2csv(
        dataset=np_latentspace_z_2,
        dataset_name=f'{out_file_name}_z_2'
    )
    return


def get_latentspace_AE_u_i_helper():
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
    print('Starting Trial 1: Get Latentspace (KVS)')

    model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_100_mae_relu_upshift'
    model_name_x = 'Model_AE_u_i_LR0_0001_x'
    model_name_y = 'Model_AE_u_i_LR0_0001_y'
    model_name_z = 'Model_AE_u_i_LR0_0001_z'
    _model_x = AE_u_i(
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

    _loader_1, _loader_2_ = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=1,
        shuffle=False
    )
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Latentspace'

    _out_file_names = [
        'kvs_latentspace_20000_NW',
        'kvs_latentspace_20000_SE',
        'kvs_latentspace_20000_SW',
        'kvs_latentspace_22000_SE',
        'kvs_latentspace_22000_SW',
        'kvs_latentspace_24000_SW',
        'kvs_latentspace_26000_NE',
        'kvs_latentspace_26000_NW',
        'kvs_latentspace_26000_SE',
        'kvs_latentspace_28000_NE',
        'kvs_latentspace_28000_NW',
        'kvs_latentspace_20000_NE',
        'kvs_latentspace_22000_NW',
        'kvs_latentspace_24000_SE',
        'kvs_latentspace_26000_SW',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE_u_i(
            loader=_loader,
            model_x=_model_x,
            model_y=_model_y,
            model_z=_model_z,
            out_file_name=f'{_out_directory}/{_out_file_names[idx]}'
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

    _model_identifier_i = f'LR{alpha_string}_i'

    print('Initializing AE_u_i model.')

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer_i = optim.Adam(_model_i.parameters(), lr=alpha)

    print('Beginning training.')
    for epoch in range(100):
        _avg_loss = 0

        for _train_loader in train_loaders:
            _loss = train_AE_u_i(
                loader=_train_loader,
                model_i=_model_i,
                optimizer_i=_optimizer_i,
                model_identifier_i=_model_identifier_i,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _avg_loss += _loss

        _avg_loss = _avg_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'[{_model_identifier_i}] Training Epoch: {epoch+1}')
        print(f'[{_model_identifier_i}] -> Avg u_i {_avg_loss:.3f}')

        _sum_loss = 0

        for _valid_loader in valid_loaders:
            _loss = valid_AE_u_i(
                loader=_train_loader,
                model_i=_model_i,
                optimizer_i=_optimizer_i,
                model_identifier_i=_model_identifier_i,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch+1
            )
            _sum_loss += _loss

        _avg_valid = _sum_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(f'[{_model_identifier_i}] Validation Epoch: {epoch+1}')
        print(f'[{_model_identifier_i}] -> Avg u_i {_avg_valid:.3f}')

    torch.save(
        _model_i.state_dict(),
        f'{_file_prefix}Model_AE_u_i_{_model_identifier_i}'
    )
    return


def trial_1_AE_mp():
    """The trial_1_AE_mp function is essentially a helper function to
    facilitate the training of multiple concurrent models via multiprocessing
    of the trial_1_AE/trial_1_AE_u_i function. Here, 6 unique models are trained
    using the 6 learning rates (_alphas) respectively. Refer to the trial_1_AE
    function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 1: AE_u_i_mp (KVS)')
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    _processes = []

    for i in range(6):
        _p = mp.Process(
            target=trial_1_AE_u_i,
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


def prediction_retriever_u_i(model_directory, model_name_x, model_name_y, model_name_z, dataset_name, save2file_prefix, save2file_name):
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

    _model_x = AE_u_i(
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
        _targs = []
        for batch_idx, (data, target) in enumerate(valid_loaders[i]):
            data = data.float().to(device=device)
            data = torch.add(data, 0.2).float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data)
                data_pred_y = _model_y(data)
                data_pred_z = _model_z(data)
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                data_pred = torch.add(
                    data_pred, -0.2).float().to(device=device)
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(target.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)

    plotPredVsTargKVS(input_1=_preds, input_2=_targs,
                      file_prefix=save2file_prefix, file_name=save2file_name)
    # plot_flow_profile(_preds, save2file_name)


if __name__ == "__main__":
    print('Starting Trial 1_mp: AE (KVS, MAE, L1Loss)')
    trial_1_AE_mp()

    '''
    _alpha = 0.0001
    _alpha_string = '0_0001'
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    trial_1_AE_u_i(_alpha, _alpha_string, _train_loaders, _valid_loaders)

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

    print('Starting Trial 1: AE_u_i (KVS + Aug, MAE, ReLU, torch.add(1.0))')
    _alpha = 0.0001
    _alpha_string = '0_0001'
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    trial_1_AE_u_i(_alpha, _alpha_string, _train_loaders, _valid_loaders)


    print('Starting Trial 1: Prediction Retriever (KVS + Aug, MAE, LReLU, AE_u_i, torch.add())')

    _model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_100_mae_relu_upshift/'
    _model_name_x = 'Model_AE_u_i_LR0_0001_x'
    _model_name_y = 'Model_AE_u_i_LR0_0001_y'
    _model_name_z = 'Model_AE_u_i_LR0_0001_z'
    _dataset_name = 'get_KVS_eval'
    _save2file_prefix = 'Model_100_relu_kvs_aug_upshift'
    _save2file_name = '22000_NW_no_std'

    prediction_retriever_u_i(
        model_directory=_model_directory,
        model_name_x=_model_name_x,
        model_name_y=_model_name_y,
        model_name_z=_model_name_z,
        dataset_name=_dataset_name,
        save2file_prefix=_save2file_prefix,
        save2file_name=_save2file_name
    )

    get_latentspace_AE_u_i_helper()
    '''
