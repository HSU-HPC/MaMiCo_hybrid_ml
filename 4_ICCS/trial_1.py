import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import AE
from utils import get_AE_loaders, losses2file, dataset2csv
from plotting import compareLossVsValid

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

    for _batch_idx, (_data, _targets) in enumerate(loader):
        _data = _data.float().to(device=device)
        _targets = _targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            _predictions = model(_data)
            _loss = criterion(_predictions.float(), _targets.float())
            _epoch_loss += _loss.item()
            print(_loss, _loss.item())
            _counter += 1

        _loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

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
            _loss = criterion(_predictions.float(), _targets.float())
            # print('Current batch loss: ', loss.item())
            _epoch_loss += _loss.item()
            _counter += 1

    _avg_loss = _epoch_loss/_counter
    return _avg_loss


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
        '3_Constituent_Hybrid_approach/Results/1_UNET_AE/Model_UNET_AE_LR0_0005'))
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
    _criterion = nn.L1Loss()
    _file_prefix = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/' + \
        '4_ICCS/Results/1_Conv_AE/'
    _model_identifier = f'LR{alpha_string}'
    print('Initializing UNET_AE model.')
    _model = AE(
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
        file_name=f'{_file_prefix}Losses_UNET_AE_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_UNET_AE_{_model_identifier}'
    )

    compareLossVsValid(
        loss_files=[
            f'{_file_prefix}Losses_UNET_AE_{_model_identifier}.csv',
            f'{_file_prefix}Valids_UNET_AE_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'UNET_AE_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_UNET_AE_{_model_identifier}'
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

    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                 '3_Constituent_Hybrid_approach/Results/1_UNET_AE/'
    _model_name = 'Model_UNET_AE_LR0_0005'
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


if __name__ == "__main__":
    # trial_1_AE_mp()
    print('Starting Trial 1: AE (KVS)')

    _alpha = 0.0001
    _alpha_string = '0_0001'
    _train_loaders, _valid_loaders = get_AE_loaders(
        data_distribution='get_KVS',
        batch_size=32,
        shuffle=True
    )

    trial_1_AE(_alpha, _alpha_string, _train_loaders, _valid_loaders)
