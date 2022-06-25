import torch
import time
import sys
import concurrent.futures
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN
from utils import get_UNET_AE_loaders, get_RNN_loaders, get_mamico_loaders, losses2file, dataset2csv
from plotting import plotAvgLoss, compareFlowProfile
from itertools import repeat

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def train_AE(loader, model, optimizer, criterion, scaler, alpha, current_epoch):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    start_time = time.time()

    # loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0
    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # loop.set_postfix(loss=loss.item())
    avg_loss = epoch_loss/counter
    duration = time.time() - start_time
    print('------------------------------------------------------------')
    print(f'{alpha} Training -> Epoch: {current_epoch}, Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def valid_AE(loader, model, criterion, scaler, alpha, current_epoch):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    start_time = time.time()

    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

    avg_loss = epoch_loss/counter
    duration = time.time() - start_time
    print('------------------------------------------------------------')
    print(f'{alpha} Validation -> Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def get_latentspace_AE(loader, model, out_file_name):

    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

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
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    #TO DO - Check proper model to load
    _model.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/Model_UNET_AE_0_001'))
    _model.eval()

    _loader_1, _ = get_mamico_loaders()
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/Latentspace_Dataset'
    _out_file_names = [
        '_0_5_T',
        '_0_5_M',
        '_0_5_B',
        '_1_0_T',
        '_1_0_M',
        '_1_0_B',
        '_2_0_T',
        '_2_0_M',
        '_2_0_B',
        '_4_0_T',
        '_4_0_M',
        '_4_0_B',
        '_3_0_T',
        '_3_0_M',
        '_3_0_B',
        '_5_0_T',
        '_5_0_M',
        '_5_0_B',
    ]
    for i in range(18):
        get_latentspace_AE(
            loader=_loader_1[i],
            model=_model,
            out_file_name=f'{_out_directory}{_out_file_names[i]}'
        )
    pass


def train_RNN(loader, model, optimizer, criterion, scaler, identifier='', current_epoch=''):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    start_time = time.time()

    # loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0
    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # loop.set_postfix(loss=loss.item())
    avg_loss = epoch_loss/counter
    duration = time.time() - start_time
    print('------------------------------------------------------------')
    print(f'{identifier} Training -> Epoch: {current_epoch}, Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def valid_RNN(loader, model, criterion, scaler, identifier='', current_epoch=''):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    start_time = time.time()

    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

    avg_loss = epoch_loss/counter
    duration = time.time() - start_time
    print('------------------------------------------------------------')
    print(f'{identifier} Validation -> Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def train_HYBRID():
    pass


def valid_HYBRID():
    pass


def trial_1_UNET_AE(_alpha, _alpha_string, _train_loader, _valid_loader):
    # _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _criterion = nn.L1Loss()
    # _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=0)
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'
    print('Initializing model.')
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _epoch_losses = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = train_AE(
            loader=_train_loader,
            model=_model,
            optimizer=_optimizer,
            criterion=_criterion,
            scaler=_scaler,
            alpha=_alpha_string,
            current_epoch=epoch+1
        )
        _epoch_losses.append(avg_loss)

    _valid_loss = valid_AE(
        loader=_valid_loader,
        model=_model,
        criterion=_criterion,
        scaler=_scaler,
        alpha=_alpha_string,
        current_epoch=0
    )
    _epoch_losses.append(_valid_loss)
    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_UNET_AE_{_alpha_string}'
    )

    plotAvgLoss(
        avg_losses=_epoch_losses,
        file_prefix=_file_prefix,
        file_name=f'UNET_AE_{_alpha_string}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_UNET_AE_{_alpha_string}'
    )

    return


def trial_1_mp():
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    # _alphas = [0.01, 0.001, 0.0001]
    # _alphas_strings = ['test1', 'test2', 'test3']
    _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=1)
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(trial_1_UNET_AE, _alphas, _alphas_strings, repeat(
            _train_loader), repeat(_valid_loader))
    '''
    processes = []
    counter = 1

    for i in range(3):
        p = mp.Process(
            target=trial_1_UNET_AE,
            args=(_alphas[i], _alpha_strings[i], _train_loader, _valid_loader,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {counter}')
        counter += 1

    for process in processes:
        process.join()
        print('Joining Process')
        return


def trial_2_RNN(_seq_length, _num_layers, _alpha, _alpha_string, _train_loader, _valid_loader):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_RNN/'
    _model_identifier = f'{_seq_length}_{_num_layers}_{_alpha_string}'
    print('Initializing model.')
    _model = RNN(
        input_size=256,
        hidden_size=256,
        num_layers=_num_layers,
        device=device
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _epoch_losses = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = train_RNN(
            loader=_train_loader,
            model=_model,
            optimizer=_optimizer,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=epoch+1
        )
        _epoch_losses.append(avg_loss)

    _valid_loss = valid_AE(
        loader=_valid_loader,
        model=_model,
        criterion=_criterion,
        scaler=_scaler,
        identifier=_model_identifier,
        current_epoch=0
    )
    _epoch_losses.append(_valid_loss)
    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_RNN_{_model_identifier}'
    )

    plotAvgLoss(
        avg_losses=_epoch_losses,
        file_prefix=_file_prefix,
        file_name=f'RNN_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_RNN_{_model_identifier}'
    )


if __name__ == "__main__":
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _t_loader, _v_loader = get_RNN_loaders(
        file_names=0, sequence_length=15)
    i = 0
    trial_2_RNN(
        _seq_length=15,
        _num_layers=2,
        _alpha=_alphas[i],
        _alpha_string=_alpha_strings[i],
        _train_loader=_t_loader,
        _valid_loader=_v_loader
    )

    pass
