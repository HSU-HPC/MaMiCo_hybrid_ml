import torch
import random
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import AE_u_x, AE_u_y, AE_u_z, RNN, GRU, LSTM, Hybrid_MD_RNN_AE_u_i
from utils import get_AE_loaders, get_RNN_loaders, dataset2csv
from plotting import plotPredVsTargKVS

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False


def train_RNN_u_i(loader_x, loader_y, loader_z, model_x, model_y, model_z, optimizer_x, optimizer_y, optimizer_z, model_identifier_x, model_identifier_y, model_identifier_z, criterion, scaler, current_epoch):
    """The train_RNN_u_i function trains the models and computes the average
    loss on the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the models to be trained.
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
          A double value indicating average training loss for the current epoch
          and model.
    """

    _epoch_loss_x = 0
    _epoch_loss_y = 0
    _epoch_loss_z = 0
    _counter = 0

    for _data_x, _targ_x in loader_x:
        # print('In training loop.')
        _data_x = _data_x.float().to(device)
        _targ_x = _targ_x.float().to(device)

        _data_y, _targ_y = next(iter(loader_y))
        _data_y = _data_y.float().to(device)
        _targ_y = _targ_y.float().to(device)

        _data_z, _targ_z = next(iter(loader_z))
        _data_z = _data_z.float().to(device)
        _targ_z = _targ_z.float().to(device)

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data_x).float().to(device=device)
            _preds_y = model_y(_data_y).float().to(device=device)
            _preds_z = model_z(_data_z).float().to(device=device)

            _loss_x = criterion(_preds_x, _targ_x)
            _loss_y = criterion(_preds_y, _targ_y)
            _loss_z = criterion(_preds_z, _targ_z)

            print(_loss_x.item())
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


def train_RNN_u_i_single(loader_x, model_x, optimizer_x, model_identifier_x, criterion, scaler, current_epoch):
    """The train_RNN_u_i_single function trains the model (!!) and computes the average
        loss on the training set.

        Args:
            loader:
              Object of PyTorch-type DataLoader to automatically feed dataset
            model:
              Object of PyTorch Module class, i.e. the models to be trained.
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
              A double value indicating average training loss for the current epoch
              and model.
        """

    _epoch_loss_x = 0
    _counter = 0

    for _data_x, _targ_x in loader_x:
        _data_x = _data_x.float().to(device)
        _targ_x = _targ_x.float().to(device)

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data_x).float().to(device=device)
            _loss_x = criterion(_preds_x, _targ_x)
            _epoch_loss_x += _loss_x.item()
            _counter += 1

        _loss_x.backward(retain_graph=True)
        optimizer_x.step()
        optimizer_x.zero_grad()

    _avg_loss_x = _epoch_loss_x/_counter
    return _avg_loss_x


def valid_RNN_u_i(loader_x, loader_y, loader_z, model_x, model_y, model_z, model_identifier_x, model_identifier_y, model_identifier_z, criterion, scaler, current_epoch):
    """The train_RNN_u_i function trains the models and computes the average
    loss on the training set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the models to be trained.
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
          A double value indicating average training loss for the current epoch
          and model.
    """

    _epoch_loss_x = 0
    _epoch_loss_y = 0
    _epoch_loss_z = 0
    _counter = 0

    for _data_x, _targ_x in loader_x:
        _data_x = _data_x.float().to(device)
        _targ_x = _targ_x.float().to(device)

        _data_y, _targ_y = next(iter(loader_y))
        _data_y = _data_y.float().to(device)
        _targ_y = _targ_y.float().to(device)

        _data_z, _targ_z = next(iter(loader_z))
        _data_z = _data_z.float().to(device)
        _targ_z = _targ_z.float().to(device)

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data_x).float().to(device=device)
            _preds_y = model_y(_data_y).float().to(device=device)
            _preds_z = model_z(_data_z).float().to(device=device)

            _loss_x = criterion(_preds_x, _targ_x)
            _loss_y = criterion(_preds_y, _targ_y)
            _loss_z = criterion(_preds_z, _targ_z)

            _epoch_loss_x += _loss_x.item()
            _epoch_loss_y += _loss_y.item()
            _epoch_loss_z += _loss_z.item()
            _counter += 1

    _avg_loss_x = _epoch_loss_x/_counter
    _avg_loss_y = _epoch_loss_y/_counter
    _avg_loss_z = _epoch_loss_z/_counter
    return _avg_loss_x, _avg_loss_y, _avg_loss_z


def valid_RNN_u_i_single(loader_x, model_x, model_identifier_x, criterion, scaler, current_epoch):
    """The valid_RNN_u_i_single function validates the models and computes the average
    loss on the validation set.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch Module class, i.e. the models to be trained.
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
          A double value indicating average training loss for the current epoch
          and model.
    """

    _epoch_loss_x = 0
    _counter = 0

    for _data_x, _targ_x in loader_x:
        _data_x = _data_x.float().to(device)
        _targ_x = _targ_x.float().to(device)

        with torch.cuda.amp.autocast():
            _preds_x = model_x(_data_x).float().to(device=device)
            _loss_x = criterion(_preds_x, _targ_x)
            _epoch_loss_x += _loss_x.item()
            _counter += 1

    _avg_loss_x = _epoch_loss_x/_counter
    return _avg_loss_x


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
        _data_0 = torch.add(_data_0, 0.2).float().to(device)
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


def prediction_retriever_latentspace_u_i(model_directory, model_name_x, model_name_y, model_name_z, dataset_name, save2file_prefix, save2file_name):
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
    _, valid_loaders = get_RNN_loaders(
            data_distribution=dataset_name,
            batch_size=1,
            shuffle=False
        )

    _, targ_loaders = get_AE_loaders(
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

    data_preds_x = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_y = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    data_preds_z = torch.zeros(1, 1, 24, 24, 24).to(device=device)
    targs = []

    for data, target in valid_loaders[0]:
        data = data.float().to(device=device)
        # print('model_x(data) -> shape: ', data.shape)
        with torch.cuda.amp.autocast():
            data_pred_x = _model_x(data, 'get_MD_output').to(device=device)
            data_preds_x = torch.cat((data_preds_x, data_pred_x), 0).to(device)

    for data, target in valid_loaders[1]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_y = _model_y(data, 'get_MD_output').to(device=device)
            data_preds_y = torch.cat((data_preds_y, data_pred_y), 0).to(device)

    for data, target in valid_loaders[2]:
        data = data.float().to(device=device)
        with torch.cuda.amp.autocast():
            data_pred_z = _model_z(data, 'get_MD_output').to(device=device)
            data_preds_z = torch.cat((data_preds_z, data_pred_z), 0).to(device)

    for data, target in targ_loaders[0]:
        with torch.cuda.amp.autocast():
            targs.append(target.cpu().detach().numpy())

    targs = np.vstack(targs)

    print('data_preds_x.shape: ', data_preds_x.shape)
    print('data_preds_y.shape: ', data_preds_y.shape)
    print('data_preds_z.shape: ', data_preds_z.shape)
    preds = torch.cat((data_preds_x, data_preds_y, data_preds_z), 1).to(device)
    preds = torch.add(preds, -0.2).float().to(device=device)
    preds = preds[1:, :, :, :, :].cpu().detach().numpy()

    plotPredVsTargKVS(input_1=preds, input_2=targs,
                      file_prefix=save2file_prefix, file_name=save2file_name)


def prediction_retriever_hybrid(model_AE_directory, model_name_x, model_name_y, model_name_z, model_RNN_directory, model_name_RNN, dataset_name, save2file_prefix, save2file_name):
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
    _, targ_loaders = get_AE_loaders(
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

    _num_layers = 2
    _seq_length = 15
    _model_RNN = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    _model_x.load_state_dict(torch.load(
        f'{model_AE_directory}/{model_name_x}', map_location='cpu'))
    _model_x.eval()
    _model_y.load_state_dict(torch.load(
        f'{model_AE_directory}/{model_name_y}', map_location='cpu'))
    _model_y.eval()
    _model_z.load_state_dict(torch.load(
        f'{model_AE_directory}/{model_name_z}', map_location='cpu'))
    _model_z.eval()
    _model_RNN.load_state_dict(torch.load(
        f'{model_RNN_directory}/{model_name_RNN}', map_location='cpu'))
    _model_RNN.eval()

    _model_Hybrid = Hybrid_MD_RNN_AE_u_i(
        device=device,
        AE_Model_x=_model_x,
        AE_Model_y=_model_y,
        AE_Model_z=_model_z,
        RNN_Model_x=_model_RNN,
        RNN_Model_y=_model_RNN,
        RNN_Model_z=_model_RNN,
        seq_length=_seq_length,
    ).to(device)

    _preds = torch.zeros(1, 3, 24, 24, 24).to(device=device)
    _targs = []

    for data, target in targ_loaders:
        data = data.float().to(device=device)
        data = torch.add(data, 0.2).float().to(device=device)
        # print('model_x(data) -> shape: ', data.shape)
        with torch.cuda.amp.autocast():
            _pred = _model_Hybrid(data)
            _preds = torch.cat((_preds, _pred), 0).to(device)
            _targs.append(target.cpu().detach().numpy())

    _preds = torch.add(_preds, -0.2).float().to(device=device)
    _preds = _preds[1:, :, :, :, :].cpu().detach().numpy()
    _targs = np.vstack(_targs)

    plotPredVsTargKVS(input_1=_preds, input_2=_targs,
                      file_prefix=save2file_prefix, file_name=save2file_name)


def trial_2_preliminary_verifications():
    print('Starting Trial 2: Prediction Retriever (KVS + Aug, MAE, ReLU, AE_u_i, torch.add())')

    _model_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_100_mae_relu_upshift/'
    _model_name_x = 'Model_AE_u_i_LR0_0001_x'
    _model_name_y = 'Model_AE_u_i_LR0_0001_y'
    _model_name_z = 'Model_AE_u_i_LR0_0001_z'
    _dataset_name = 'get_KVS_eval'
    _save2file_prefix = 'Model_100_relu_kvs_aug_upshift_latentspace'
    _save2file_name = '22000_NW_no_std'

    prediction_retriever_latentspace_u_i(
        model_directory=_model_directory,
        model_name_x=_model_name_x,
        model_name_y=_model_name_y,
        model_name_z=_model_name_z,
        dataset_name=_dataset_name,
        save2file_prefix=_save2file_prefix,
        save2file_name=_save2file_name
    )


def trial_2_train_RNN():
    """The analysis_2_Couette_RNN function trains an RNN model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        NONE

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    print('Starting Trial 2: Train RNN (Random, MAE, RNN_u_i)')

    print('Initializing RNN datasets.')
    _train_x, _train_y, _train_z, _valid_x, _valid_y, _valid_z = get_RNN_loaders(
        data_distribution='get_KVS')

    print('Initializing training parameters.')

    _criterion = nn.L1Loss()
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _alpha_string = '1e-3'
    _alpha = 1e-3
    _num_layers = 2
    _seq_length = 15
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing RNN model.')

    _model_x = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_x = optim.Adam(_model_x.parameters(), lr=_alpha)
    _model_identifier_x = _model_identifier + 'x'

    _model_y = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_y = optim.Adam(_model_y.parameters(), lr=_alpha)
    _model_identifier_y = _model_identifier + 'y'

    _model_z = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_z = optim.Adam(_model_z.parameters(), lr=_alpha)
    _model_identifier_z = _model_identifier + 'z'

    _scaler = torch.cuda.amp.GradScaler()

    print('Beginning training.')
    for epoch in range(15):
        _avg_loss_x = 0
        _avg_loss_y = 0
        _avg_loss_z = 0
        for idx, _train_loader in enumerate(_train_x):
            loss_x, loss_y, loss_z = train_RNN_u_i(
                loader_x=_train_x[idx],
                loader_y=_train_y[idx],
                loader_z=_train_z[idx],
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
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
            _avg_loss_y += loss_y
            _avg_loss_z += loss_z
        _avg_loss_x = _avg_loss_x/len(_train_x)
        _avg_loss_y = _avg_loss_y/len(_train_x)
        _avg_loss_z = _avg_loss_z/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Training Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')
        print(f'-> Avg u_y {_avg_loss_y:.3f}')
        print(f'-> Avg u_z {_avg_loss_z:.3f}')

        _avg_loss_x = 0
        _avg_loss_y = 0
        _avg_loss_z = 0

        for idx, _valid_loader in enumerate(_valid_x):
            loss_x, loss_y, loss_z = valid_RNN_u_i(
                loader_x=_train_x[idx],
                loader_y=_train_y[idx],
                loader_z=_train_z[idx],
                model_x=_model_x,
                model_y=_model_y,
                model_z=_model_z,
                model_identifier_x=_model_identifier_x,
                model_identifier_y=_model_identifier_y,
                model_identifier_z=_model_identifier_z,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
            _avg_loss_y += loss_y
            _avg_loss_z += loss_z
        _avg_loss_x = _avg_loss_x/len(_train_x)
        _avg_loss_y = _avg_loss_y/len(_train_x)
        _avg_loss_z = _avg_loss_z/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')
        print(f'-> Avg u_y {_avg_loss_y:.3f}')
        print(f'-> Avg u_z {_avg_loss_z:.3f}')

    torch.save(
        _model_x.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_x}'
    )
    torch.save(
        _model_y.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_y}'
    )
    torch.save(
        _model_z.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_z}'
    )


def trial_2_train_RNN_single():
    """The trial_2_train_RNN_single function trains an RNN model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        NONE

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    print('Starting Trial 2: Train RNN (KVS, MAE, single RNN)')

    print('Initializing RNN datasets.')
    _train_x, _train_y, _train_z, _valid_x, _valid_y, _valid_z = get_RNN_loaders(
        data_distribution='get_KVS')

    print('Initializing training parameters.')

    _criterion = nn.L1Loss()
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _alpha_string = '1e-5'
    _alpha = 1e-5
    _num_layers = 1
    _seq_length = 25
    _model_identifier = f'RNN_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing RNN model.')

    _model_x = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_x = optim.Adam(_model_x.parameters(), lr=_alpha)
    _model_identifier_x = _model_identifier + '_x'

    _scaler = torch.cuda.amp.GradScaler()

    print('Beginning training.')
    for epoch in range(100):
        _avg_loss_x = 0
        for idx, _train_loader in enumerate(_train_x):
            loss_x = train_RNN_u_i_single(
                loader_x=_train_x[idx],
                model_x=_model_x,
                optimizer_x=_optimizer_x,
                model_identifier_x=_model_identifier_x,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
        _avg_loss_x = _avg_loss_x/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Training Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')

        _avg_loss_x = 0

        for idx, _valid_loader in enumerate(_valid_x):
            loss_x = valid_RNN_u_i_single(
                loader_x=_train_x[idx],
                model_x=_model_x,
                model_identifier_x=_model_identifier_x,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
        _avg_loss_x = _avg_loss_x/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')

    torch.save(
        _model_x.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_x}'
    )


def trial_2_train_LSTM_single():
    """The trial_2_train_LSTM_single function trains an LSTM model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        NONE

    Returns:
        NONE:
          This function documents model progress by saving results to file and
          creating meaningful plots.
    """
    print('Starting Trial 2: Train LSTM (KVS + Aug, MAE, single LSTM)')

    print('Initializing RNN datasets.')
    _train_x, _train_y, _train_z, _valid_x, _valid_y, _valid_z = get_RNN_loaders(
        data_distribution='get_KVS')

    print('Initializing training parameters.')

    _criterion = nn.L1Loss()
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/'
    _alpha_string = '1e-5'
    _alpha = 1e-5
    _num_layers = 1
    _seq_length = 25
    _model_identifier = f'LSTM_LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing LSTM model.')

    _model_x = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)
    _optimizer_x = optim.Adam(_model_x.parameters(), lr=_alpha)
    _model_identifier_x = _model_identifier + 'x'

    _scaler = torch.cuda.amp.GradScaler()

    print('Beginning training.')
    for epoch in range(100):
        _avg_loss_x = 0
        for idx, _train_loader in enumerate(_train_x):
            loss_x = train_RNN_u_i_single(
                loader_x=_train_x[idx],
                model_x=_model_x,
                optimizer_x=_optimizer_x,
                model_identifier_x=_model_identifier_x,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
        _avg_loss_x = _avg_loss_x/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Training Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')

        _avg_loss_x = 0

        for idx, _valid_loader in enumerate(_valid_x):
            loss_x = valid_RNN_u_i_single(
                loader_x=_train_x[idx],
                model_x=_model_x,
                model_identifier_x=_model_identifier_x,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=epoch
            )
            _avg_loss_x += loss_x
        _avg_loss_x = _avg_loss_x/len(_train_x)
        print('------------------------------------------------------------')
        print(f'Validation Epoch: {epoch+1}')
        print(f'-> Avg u_x {_avg_loss_x:.3f}')

    torch.save(
        _model_x.state_dict(),
        f'{_file_prefix}Model_{_model_identifier_x}'
    )


def trial_2_RNN_single_verification():
    print('Starting Trial 2: Prediction Retriever (KVS + Aug, MAE, ReLU, AE_u_i, torch.add())')

    _model_AE_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/1_Conv_AE/kvs_aug_100_mae_relu_upshift/'
    _model_RNN_directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/Results/2_RNN/kvs_aug_15_mae_single_RNN/'
    _model_name_x = 'Model_AE_u_i_LR0_0001_x'
    _model_name_y = 'Model_AE_u_i_LR0_0001_y'
    _model_name_z = 'Model_AE_u_i_LR0_0001_z'
    _model_name_RNN = 'Model_LR1e-3_Lay2_Seq15x'
    _dataset_name = 'get_KVS_eval'
    _save2file_prefix = 'Model_100_relu_kvs_aug_upshift_Hybrid_RNN_Piet_2'
    _save2file_name = '22000_NW_no_std'

    prediction_retriever_hybrid(
        model_AE_directory=_model_AE_directory,
        model_name_x=_model_name_x,
        model_name_y=_model_name_y,
        model_name_z=_model_name_z,
        model_RNN_directory=_model_RNN_directory,
        model_name_RNN=_model_name_RNN,
        dataset_name=_dataset_name,
        save2file_prefix=_save2file_prefix,
        save2file_name=_save2file_name
    )


if __name__ == "__main__":
    # trial_2_train_LSTM_single()
    trial_2_RNN_single_verification()

    pass
