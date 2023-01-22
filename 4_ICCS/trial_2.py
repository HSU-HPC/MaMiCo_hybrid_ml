import torch
import random
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from model import AE_u_x, AE_u_y, AE_u_z
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


if __name__ == "__main__":
    get_latentspace_AE_u_i_helper()
    '''
    print('Starting Trial 2: Prediction Retriever (KVS + Aug, MAE, LReLU, AE_u_i, torch.add())')

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
    '''
