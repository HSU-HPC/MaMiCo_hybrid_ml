"""trial_1

This script focuses on training the convolutional autoencoder as used in the
triple model approach. Afterwards, the latentspaces are retrieved from the
trained model. Plots are created.
"""

import torch
import random
import copy
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import platform
import time
from model import AE_u_i
from utils import get_AE_loaders, get_RNN_loaders, dataset2csv, mlready2dataset
from plotting import plot_flow_profile, plotPredVsTargKVS

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


def get_latentspace_AE_u_i(loader, model_i, out_file_name):
    """The get_latentspace_AE function extracts the model-specific latentspace
    for a given dataset and saves it to file.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the model from which to extract
          the latentspace.
        out_file_name:
          A string containing the name of the file that the latentspace should
          be saved to.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          latentspace to file.
    """
    latentspace_x = []
    latentspace_y = []
    latentspace_z = []

    for batch_idx, (_data, _) in enumerate(loader):
        t, c, h, d, w = _data.shape
        _data = torch.add(_data, 1.0).float().to(device)
        _data_x = _data[:, 0, :, :, :]
        _data_y = _data[:, 1, :, :, :]
        _data_z = _data[:, 2, :, :, :]

        with torch.cuda.amp.autocast():
            bottleneck_x = model_i(_data_x,  y='get_bottleneck')
            bottleneck_y = model_i(_data_y,  y='get_bottleneck')
            bottleneck_z = model_i(_data_z,  y='get_bottleneck')
            latentspace_x.append(bottleneck_x.cpu().detach().numpy())
            latentspace_y.append(bottleneck_y.cpu().detach().numpy())
            latentspace_z.append(bottleneck_z.cpu().detach().numpy())

    np_latentspace_x = np.vstack(latentspace_x)
    np_latentspace_y = np.vstack(latentspace_y)
    np_latentspace_z = np.vstack(latentspace_z)

    dataset2csv(
        dataset=np_latentspace_x,
        dataset_name=f'{out_file_name}_x'
    )
    dataset2csv(
        dataset=np_latentspace_y,
        dataset_name=f'{out_file_name}_y'
    )
    dataset2csv(
        dataset=np_latentspace_z,
        dataset_name=f'{out_file_name}_z'
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

    model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE'
    model_name_i = 'Model_AE_u_i_LR0_001_i'
    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_i.eval()

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
        'kvs_latentspace_22000_NE',
        'kvs_latentspace_22000_SE',
        'kvs_latentspace_22000_SW',
        'kvs_latentspace_24000_NE',
        'kvs_latentspace_24000_NW',
        'kvs_latentspace_24000_SE',
        'kvs_latentspace_24000_SW',
        'kvs_latentspace_26000_NE',
        'kvs_latentspace_26000_NW',
        'kvs_latentspace_26000_SW',
        'kvs_latentspace_28000_NE',
        'kvs_latentspace_28000_NW',
        'kvs_latentspace_28000_SE',
        'kvs_latentspace_20000_NE',
        'kvs_latentspace_22000_NW',
        'kvs_latentspace_26000_SE',
        'kvs_latentspace_28000_SW',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE_u_i(
            loader=_loader,
            model_i=_model_i,
            out_file_name=f'{_out_directory}/{_out_file_names[idx]}'
        )


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
    for epoch in range(1):
        _avg_loss = 0
        print('Hardware: ', platform.processor())

        start = time.time()

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

        end = time.time()
        print('Duration of one ML Calculation = 1 Coupling Cycle: ', end - start)

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


def trial_1_AE_u_i_mp():
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
    _alphas = [0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_001', '0_0005', '0_0001', '0_00005']
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    _processes = []

    for i in range(1):
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


def prediction_retriever_u_i(model_directory, model_name_i, dataset_name, save2file_name_1, save2file_name_2):
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
    _1_train_loaders, _2_valid_loaders = get_AE_loaders(
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
    _model_y = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_z = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_x.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_x.eval()
    _model_y.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_y.eval()
    _model_z.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_z.eval()

    for train_loader in _1_train_loaders:
        _preds = []
        _targs = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device=device)
            data = torch.add(data, 1.0).float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data[:, 0, :, :, :])
                data_pred_y = _model_y(data[:, 1, :, :, :])
                data_pred_z = _model_z(data[:, 2, :, :, :])
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                data_pred = torch.add(
                    data_pred, -1.0).float().to(device=device)
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(target.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)
    _lbm_1 = np.loadtxt(
        f'dataset_mlready/01_clean_lbm/kvs_{save2file_name_1}_lbm.csv', delimiter=";")
    _lbm_1 = _lbm_1.reshape(1000, 3)
    plotPredVsTargKVS(input_pred=_preds, input_targ=_targs,
                      input_lbm=_lbm_1, file_name=save2file_name_1)

    for valid_loader in _2_valid_loaders:
        _preds = []
        _targs = []
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.float().to(device=device)
            data = torch.add(data, 1.0).float().to(device=device)
            with torch.cuda.amp.autocast():
                data_pred_x = _model_x(data[:, 0, :, :, :])
                data_pred_y = _model_y(data[:, 1, :, :, :])
                data_pred_z = _model_z(data[:, 2, :, :, :])
                data_pred = torch.cat(
                    (data_pred_x, data_pred_y, data_pred_z), 1).to(device)
                data_pred = torch.add(
                    data_pred, -1.0).float().to(device=device)
                _preds.append(data_pred.cpu().detach().numpy())
                _targs.append(target.cpu().detach().numpy())
        _preds = np.vstack(_preds)
        _targs = np.vstack(_targs)
    _lbm_2 = np.loadtxt(
        f'dataset_mlready/01_clean_lbm/kvs_{save2file_name_2}_lbm.csv', delimiter=";")
    _lbm_2 = _lbm_2.reshape(1000, 3)
    plotPredVsTargKVS(input_pred=_preds, input_targ=_targs,
                      input_lbm=_lbm_2, file_name=save2file_name_2)


def prediction_retriever_latentspace_u_i(model_directory, model_name_i, dataset_name, save2file_name):
    """The prediction_retriever function_latentspace_u_i is used to evaluate
    correctness of extracted latentspaces. This is done by loading the trained
    AE_u_i model, feeding it with the extracted latentspaces, saving the corresponding predictions for later
    visual comparison.

    Args:
        model_directory:

        model_name_i:

        dataset_name:

        save2file_prefix:

        save2file_name:

    Returns:
        NONE
    """
    _t_x, _t_y, _t_z, _v_x, _v_y, _v_z = get_RNN_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _trains, _valids = get_AE_loaders(
            data_distribution=dataset_name,
            batch_size=32,
            shuffle=False
        )

    _model_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _model_i.load_state_dict(torch.load(
        f'{model_directory}/{model_name_i}', map_location='cpu'))
    _model_i.eval()

    data_preds_x = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_y = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_z = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    targs = []

    for data, target in _t_x[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_x = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_x = torch.cat((data_preds_x, data_pred_x), 0).to(device)

    for data, target in _t_y[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_y = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_y = torch.cat((data_preds_y, data_pred_y), 0).to(device)

    for data, target in _t_z[0]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_z = _model_i(data, 'get_MD_output').to(device=device)
            data_preds_z = torch.cat((data_preds_z, data_pred_z), 0).to(device)

    for data, target in _trains[0]:
        with torch.cuda.amp.autocast():
            target = target.cpu().detach().numpy()
            print(target.shape)
            targs.append(target)

    targs = np.vstack(targs)

    print('data_preds_x.shape: ', data_preds_x.shape)
    print('data_preds_y.shape: ', data_preds_y.shape)
    print('data_preds_z.shape: ', data_preds_z.shape)
    preds = torch.cat((data_preds_x, data_preds_y, data_preds_z), 1).to(device)
    preds = torch.add(preds, -1.0).float().to(device=device)
    preds = preds[1:, :, :, :, :].cpu().detach().numpy()

    plotPredVsTargKVS(input_pred=preds, input_targ=targs,
                      file_name=save2file_name)


def fig_maker_1(id):
    _directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/KVS/Validation/'
    _id = id  # '28000_SW'
    _file_name = f'clean_kvs_combined_domain_init_{_id}.csv'
    _model_directory = 'Results/1_Conv_AE'
    _model_name_i = 'Model_AE_u_i_LR0_001_i'

    _dataset = torch.from_numpy(mlready2dataset(
        f'{_directory}{_file_name}')[:, :, 1:-1, 1:-1, 1:-1])
    _targs = copy.deepcopy(_dataset)
    _dataset = torch.add(_dataset, 1.0).float().to(device)

    _model_u_i = AE_u_i(
        device=device,
        in_channels=1,
        out_channels=1,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_u_i.load_state_dict(torch.load(
        f'{_model_directory}/{_model_name_i}', map_location='cpu'))
    _model_u_i.eval()

    _preds_x = _model_u_i(_dataset[:, 0, :, :, :])
    _preds_y = _model_u_i(_dataset[:, 1, :, :, :])
    _preds_z = _model_u_i(_dataset[:, 2, :, :, :])

    _preds = torch.cat((_preds_x, _preds_y, _preds_z), 1).to(device)
    _preds = torch.add(_preds, -1.0).float().to(device)
    _preds = _preds.cpu().detach().numpy()
    _targs = _targs.numpy()
    plot_flow_profile(
        np_datasets=[_targs, _preds],
        dataset_legends=['MD', 'Autoencoder'],
        save2file=f'{_id}_Fig_Maker_5_a_ConvAE_vs_MD'
    )
    pass


if __name__ == "__main__":
    '''
    _model_directory = 'Results/1_Conv_AE'
    _model_name_i = 'Model_AE_u_i_LR0_001_i'
    _dataset_name = 'get_KVS_eval'
    _save2file_name = 'latentspace_validation_20k_NE'

    prediction_retriever_latentspace_u_i(
        model_directory=_model_directory,
        model_name_i=_model_name_i,
        dataset_name=_dataset_name,
        save2file_name=_save2file_name)

    _ids = ['20000_NE', '22000_NW', '26000_SE', '28000_SW']
    for _id in _ids:
        fig_maker_1(id=_id)
    '''
    trial_1_AE_u_i_mp()
