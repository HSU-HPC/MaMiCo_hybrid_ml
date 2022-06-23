import torch
import time
import sys
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import UNET_AE, Hybrid_MD_RNN_UNET, Hybrid_MD_GRU_UNET, Hybrid_MD_LSTM_UNET, resetPipeline
from utils import get_UNET_AE_loaders, get_mamico_loaders, losses2file, checkUserModelSpecs, dataset2csv
from plotting import plotAvgLoss, compareFlowProfile
from tqdm import tqdm

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def train_AE(loader, model, optimizer, criterion, scaler, current_epoch):
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
    print('                         Training')
    print('   ')
    print(f'                         Epoch: {current_epoch}')
    print(f'                      Avg Loss: {avg_loss:.3f}')
    print(f'                      Duration: {duration:.3f}')
    print('------------------------------------------------------------')
    return avg_loss


def valid_AE(loader, model, criterion, scaler, current_epoch):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    start_time = time.time()

    loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss/counter
    duration = time.time() - start_time
    print('------------------------------------------------------------')
    print('                        Validation')
    print(f'                      Avg Loss: {avg_loss:.3f}')
    print(f'                      Duration: {duration:.3f}')
    print('------------------------------------------------------------')
    return avg_loss


def trial_1():
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)

    _alpha = 0.002
    _alpha_string = '0_002'
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _criterion = nn.L1Loss()
    _scaler = torch.cuda.amp.GradScaler()
    _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=1)
    _epoch_losses = []
    for epoch in range(20):
        avg_loss = train_AE(
            loader=_train_loader,
            model=_model,
            optimizer=_optimizer,
            criterion=_criterion,
            scaler=_scaler,
            current_epoch=epoch+1
        )
        _epoch_losses.append(avg_loss)
        print(f'                          Epoch {epoch}')
        print(f'                      Avg Loss: {avg_loss:.3f}')
        print('------------------------------------------------------------')

    _valid_loss = valid_AE(
        loader=_valid_loader,
        model=_model,
        criterion=_criterion,
        scaler=_scaler,
        current_epoch=0
    )
    _epoch_losses.append(_valid_loss)
    losses2file(losses=_epoch_losses,
                filename=f'Losses_UNET_AE_{_alpha_string}')

    plotAvgLoss(avg_losses=_epoch_losses, file_name=f'UNET_AE_{_alpha_string}')
    return


if __name__ == "__main__":
    trial_1()
