from utils import get_mamico_loaders, losses2file, userModelSpecs
import time
from model import Hybrid_MD_RNN_UNET, Hybrid_MD_GRU_UNET, Hybrid_MD_LSTM_UNET
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def train_hybrid(loader, model, optimizer, criterion, scaler, current_epoch):
    # BRIEF: The train function will complete one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    losses = []
    # @losses - container for each individually calculated loss
    counter = 0
    # @counter - running counter to track number of batches in epoch
    max_loss = 0
    # @max_loss - stores the largest loss in this epoch
    time_buffer = 0
    # @time_buffer - used to track time at which max_loss occurs

    for batch_idx, (data, targets) in enumerate(tqdm(loader, position=0, leave=True)):
        print(" \n")

        data = data.float().squeeze(1).to(device)
        targets = targets.float().to(device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

        # backward
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # Check for max error
        counter += 1
        if loss > max_loss:
            max_loss = loss
            time_buffer = counter

    # Saving error values
    max_loss = max(losses)
    min_loss = min(losses)
    final_loss = losses[-1]
    average_loss = sum(losses)/len(losses)
    print('------------------------------------------------------------')
    print(f'Current epoch: {current_epoch}')
    print('Losses for the first 12 inputs:')
    print(f'[0]: {losses[0]:.7f}, [1]: {losses[1]:.7f}, [2]: {losses[2]:.7f}')
    print(f'[3]: {losses[3]:.7f}, [4]: {losses[4]:.7f}, [5]: {losses[5]:.7f}')
    print(f'[6]: {losses[6]:.7f}, [7]: {losses[7]:.7f}, [8]: {losses[8]:.7f}')
    print(f'[9]: {losses[9]:.7f}, [10]: {losses[9]:.7f}, [11]: {losses[9]:.7f}.')
    print(
        f'Max loss at t={time_buffer}: {max_loss:.7f}, Min loss: {min_loss:.7f}')
    print(f'Final loss: {final_loss:.7f}, Average loss: {average_loss:.7f}.')
    print('------------------------------------------------------------')
    return (max_loss, min_loss, final_loss, average_loss)


def valid_hybrid():
    pass


def test_hybrid():
    pass


def training_factory(user_input):
    # BRIEF: The training factory enables to train each hybrid model with
    # specified hyperparameters.
    # PARAMETERS:
    # user_input - this is the list returned by userModelSpecs() that prompts
    # the user to define the training hyperparameters. It consists of:
    # [model_name, rnn_layers, hid_size, learning_rate]
    # _model_names =['Hybrid_MD_RNN_UNET', 'Hybrid_MD_GRU_UNET', 'Hybrid_MD_LSTM_UNET']
    _model_name, _rnn_layer, _hid_size, _learning_rate = user_input
    # @above - unpacking user_input
    _model_names = ['Hybrid_MD_RNN_UNET',
                    'Hybrid_MD_GRU_UNET', 'Hybrid_MD_LSTM_UNET']
    # @_model_names - model names as strings for later file naming
    _rnn_layers = [2, 3, 4]
    # @_rnn_layers - container to hold the number of rnn layers deemed worth testing
    _hid_sizes = [256, 512, 768]
    # @_hid_sizes - container to hold the number of nodes per hidden layer deemed worth testing
    _learning_rates = [0.0001, 0.00005, 0.00001]
    # @_learning_rates - container to hold the learning rates deemed worth testing
    _criterion = nn.L1Loss()
    # @_criterion - initializes the loss function to the well known Mean Absolute Error (MAE)
    # _batch_size = 1
    # @_batch_size - other batch sizes are not possible. Refer to model description for more intuition
    _num_epochs = 15
    # @_num_epochs - the amount of times the model will train with each dataset
    _train_loaders = get_mamico_loaders()
    # @_train_loaders - container to hold the dataloaders for each dataset
    _scaler = torch.cuda.amp.GradScaler()
    # @_scaler - @@@@@@@@@@@@
    _max_losses = []
    # @_max_losses - container to hold the maximum loss from each epoch&dataloader
    _min_losses = []
    # @_min_losses - container to hold the minimum loss from each epoch&dataloader
    _average_losses = []
    # @_average_losses - container to hold the average loss from each epoch&dataloader

    # Hybrid_MD_RNN_UNET
    if _model_name == 0:
        _model = Hybrid_MD_RNN_UNET(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True),
            RNN_in_size=256,
            RNN_hid_size=_hid_sizes[_hid_size],
            RNN_lay=_rnn_layers[_rnn_layer]
        ).to(device)

    # Hybrid_MD_GRU_UNET
    if _model_name == 1:
        _model = Hybrid_MD_GRU_UNET(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True),
            RNN_in_size=256,
            RNN_hid_size=_hid_sizes[_hid_size],
            RNN_lay=_rnn_layers[_rnn_layer]
        ).to(device)

    # Hybrid_MD_GRU_UNET
    if _model_name == 2:
        _model = Hybrid_MD_LSTM_UNET(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True),
            RNN_in_size=256,
            RNN_hid_size=_hid_sizes[_hid_size],
            RNN_lay=_rnn_layers[_rnn_layer]
        ).to(device)

    _optimizer = optim.Adam(_model.parameters(),
                            lr=_learning_rates[_learning_rate])

    for _epoch in range(_num_epochs):
        for _train_loader in _train_loaders:
            _interim_loss = train_hybrid(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                current_epoch=_epoch
            )
            _max_losses.append(_interim_loss[0])
            _min_losses.append(_interim_loss[1])
            _average_losses.append(_interim_loss[3])

    losses2file(_average_losses,
                f'{_model_names[_model_name]}_average_MAE_{_rnn_layer}_{_hid_size}_{_learning_rate}')
    losses2file(
        _max_losses, f'{_model_names[_model_name]}_max_MAE_{_rnn_layer}_{_hid_size}_{_learning_rate}')
    losses2file(
        _min_losses, f'{_model_names[_model_name]}_min_MAE_{_rnn_layer}_{_hid_size}_{_learning_rate}')
    # @losses2file is used to evaluate the development of the models loss.
    # Here, not only the average loss is tracked, but also the min and max
    # losses in order to track the deviation from the average.
    model_performance(
        model_name=_model_name,
        rnn_layer=_rnn_layer,
        hid_size=_hid_size,
        learning_rate=_learning_rate,
        max_losses=_max_losses,
        min_losses=_min_losses,
        average_losses=_average_losses,
        epochs=_num_epochs)
    return

    torch.save(_model.state_dict(
    ), f'{_model_names[_model_name]}_{_rnn_layer}_{_hid_size}_{_learning_rate}')


def model_performance(model_name, rnn_layer, hid_size, learning_rate, max_losses=0, min_losses=0, average_losses=0, epochs=0):
    print('------------------------------------------------------------')
    print('                     Model Performance                      ')
    print('------------------------------------------------------------')
    print(f'Name: {model_name}')
    print(f'Num RNN Layers: {rnn_layer}')
    print(f'Num Nodes per Hidden Layer: {hid_size}')
    print(f'Learning rate: {learning_rate}')

    if epochs != 0:
        print(f'Num epochs: {epochs}')
        for i in range(len(max_losses)):
            if i < 9:
                print(
                    f'Counter: 0{i+1}, Max loss: {max_losses[i]:.7f}, Min loss: {min_losses[i]:.7f}, Average loss: {average_losses[i]:.7f}.')
            else:
                print(
                    f'Counter: {i+1}, Max loss: {max_losses[i]:.7f}, Min loss: {min_losses[i]:.7f}, Average loss: {average_losses[i]:.7f}.')
    print('------------------------------------------------------------')


def main():
    training_factory(userModelSpecs())


if __name__ == "__main__":
    main()
