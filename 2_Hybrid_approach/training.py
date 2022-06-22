import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import Hybrid_MD_RNN_UNET, Hybrid_MD_GRU_UNET, Hybrid_MD_LSTM_UNET, resetPipeline
import time
from utils import get_mamico_loaders, losses2file, checkUserModelSpecs, dataset2csv
from plotting import plotMinMaxAvgLoss
import concurrent.futures

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def train_hybrid(loader, model, optimizer, criterion, scaler, current_epoch):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    losses = []
    # @losses - container for each individually calculated loss
    counter = 0
    # @counter - running counter to track number of batches in epoch. This is
    # essentially a way to track the timestep.
    max_loss = 0
    # @max_loss - stores the largest loss in this epoch
    time_buffer = 0
    # @time_buffer - used to track time at which max_loss occurs

    for batch_idx, (data, targets) in enumerate(loader):
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

        if counter < 10:
            num = '000'
        elif counter < 100:
            num = '00'
        elif counter < 1000:
            num = '0'

        print(f'Progress: {num}{counter}/1000     Error: {loss:.7f}')

    # Saving error values
    max_loss = max(losses)
    min_loss = min(losses)
    final_loss = losses[-1]
    average_loss = sum(losses)/len(losses)
    print('------------------------------------------------------------')
    # print(f'Current epoch: {current_epoch}')
    print('Losses for the first 12 inputs:')
    print(f'[0]: {losses[0]:.7f}, [1]: {losses[1]:.7f}, [2]: {losses[2]:.7f}')
    print(f'[3]: {losses[3]:.7f}, [4]: {losses[4]:.7f}, [5]: {losses[5]:.7f}')
    print(f'[6]: {losses[6]:.7f}, [7]: {losses[7]:.7f}, [8]: {losses[8]:.7f}')
    print(f'[9]: {losses[9]:.7f}, [10]: {losses[9]:.7f}, [11]: {losses[9]:.7f}.')
    print(
        f'Max loss at t={time_buffer}: {max_loss:.7f}, Min loss: {min_loss:.7f}')
    print(f'Final loss: {final_loss:.7f}, Average loss: {average_loss:.7f}.')
    print('------------------------------------------------------------')
    return [max_loss, min_loss, final_loss, average_loss]


def valid_hybrid(loader, model, criterion, scaler):
    # BRIEF: The valid function completes an epoch using the validation
    # loader WITHOUT updating the model. It is used as a performance metric.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be validated
    # criterion - the loss function applied to quantify the error
    # scaler -

    losses = []
    # @losses - container for each individually calculated loss
    counter = 0
    # @counter - running counter to track number of batches in epoch. This is
    # essentially a way to track the timestep
    sample_times = [0, 25, 50, 100, 200, 400, 800, 998]
    max_loss = 0
    # @max_loss - stores the largest loss in this epoch
    time_buffer = 0
    # @time_buffer - used to track time at which max_loss occurs
    model_preds = []
    # @model_preds - container for specific predictions to be used for plotting
    model_targs = []
    # @model_targs - container for specific targets to be used for plotting

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().squeeze(1).to(device)
        targets = targets.float().to(device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

        if counter in sample_times:
            model_preds.append(scores.cpu().detach().numpy())
            model_targs.append(targets.cpu().detach().numpy())

        # Check for max error
        counter += 1
        if loss > max_loss:
            max_loss = loss
            time_buffer = counter

        if counter < 10:
            num = '000'
        elif counter < 100:
            num = '00'
        elif counter < 1000:
            num = '0'

        print(f'Progress: {num}{counter}/1000     Error: {loss:.7f}')

    # Saving error values
    max_loss = max(losses)
    min_loss = min(losses)
    avg_loss = sum(losses)/len(losses)

    print('------------------------------------------------------------')
    print('                         Validation')
    print('------------------------------------------------------------')
    print(
        f'Average error: {avg_loss:.7f}. Max error: {max(losses):.7f} at time: {time_buffer}')

    return [max_loss, min_loss, avg_loss, np.stack(model_preds), np.stack(model_targs)]


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

    _file_suffix = f'{_model_name+1}_{_rnn_layer+1}_{_hid_size+1}_{_learning_rate+1}'
    # @above - unpacking user_input
    # _model_names = ['Hybrid_MD_RNN_UNET', 'Hybrid_MD_GRU_UNET', 'Hybrid_MD_LSTM_UNET']
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
    _num_epochs = 1
    # @_num_epochs - the amount of times the model will train with each dataset
    _train_loaders, _valid_loaders = get_mamico_loaders()
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

    # Hybrid_MD_LSTM_UNET
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
    # Training
    for _epoch in range(_num_epochs):
        for _train_loader in _train_loaders:
            resetPipeline(_model)
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

    losses2file(_average_losses, f'Avg_Error_Training_{_file_suffix}')
    losses2file(_max_losses, f'Max_Error_Training_{_file_suffix}')
    losses2file(_min_losses, f'Min_Error_Training_{_file_suffix}')
    # @losses2file is used to evaluate the development of the models loss.
    # Here, not only the average loss is tracked, but also the min and max
    # losses in order to track the deviation from the average.

    plotMinMaxAvgLoss(
        min_losses=_min_losses,
        avg_losses=_average_losses,
        max_losses=_max_losses,
        file_name=_file_suffix
    )
    # @plotMinMaxAvgLoss is used to automatically graph the learning
    # process via losses. Refer to function definition for more details.

    _max_valid_losses = []
    _min_valid_losses = []
    _avg_valid_losses = []
    _counter = 1

    # Validation
    for _valid_loader in _valid_loaders:
        resetPipeline(_model)
        _results = valid_hybrid(
            loader=_valid_loader,
            model=_model,
            criterion=_criterion,
            scaler=_scaler
        )
        _max_valid_losses.append(_results[0])
        _min_valid_losses.append(_results[1])
        _avg_valid_losses.append(_results[2])
        dataset2csv(
            dataset=_results[3],
            model_descriptor=_file_suffix,
            dataset_name='preds',
            counter=_counter)
        dataset2csv(
            dataset=_results[4],
            model_descriptor=_file_suffix,
            dataset_name='targs',
            counter=_counter)
        _counter += 1

    losses2file(_max_valid_losses, f'Max_Error_Validation_{_file_suffix}')
    losses2file(_min_valid_losses, f'Min_Error_Validation_{_file_suffix}')
    losses2file(_avg_valid_losses, f'Avg_Error_Validation_{_file_suffix}')

    model_performance(
        model_name=_model_name,
        rnn_layer=_rnn_layer,
        hid_size=_hid_size,
        learning_rate=_learning_rate,
        max_losses=_max_losses,
        min_losses=_min_losses,
        average_losses=_average_losses,
        epochs=_num_epochs)
    # @model_performance is only used as a means of printing an easy to read
    # performance overview to the terminal.

    torch.save(_model.state_dict(), f'Model_{_file_suffix}')
    return


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
    # BRIEF: This allows the user to use command line arguments to be used as
    # arguments for the training process.
    #
    # PARAMETERS:
    #
    # _model_name, _rnn_layer, _hid_size, _learning_rate = user_input. Note that
    # each argument must be an integer value between 1 and 3. The following list
    # describes the corresponding meaning:
    #
    # _model_name -    1) Hybrid_MD_RNN_UNET
    #                  2) Hybrid_MD_GRU_UNET
    #                  3) Hybrid_MD_LSTM_UNET
    #
    # _rnn_layer -     1) 2
    #                  2) 3
    #                  3) 4
    #
    # _hid_size -      1) 256
    #                  2) 512
    #                  3) 768
    #
    # _learning_rate - 1) 0.0001
    #                  2) 0.00005
    #                  3) 0.00001

    _user_input = sys.argv[1:]
    _valid_input = checkUserModelSpecs(_user_input)

    if(_valid_input):
        print('Input is valid.')
        # for i in _user_input:
        for i in range(len(_user_input)):
            # print(f'before change to index: {_user_input[i]}')
            _user_input[i] = _user_input[i] - 1
            # print(f'after change to index: {_user_input[i]}')
            # @above - for loop to turn user input into proper indices

        training_factory(_user_input)

    else:
        print('Input is invalid.')


def load_model():
    model = Hybrid_MD_GRU_UNET(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True),
        RNN_in_size=256,
        RNN_hid_size=256,
        RNN_lay=2
    ).to(device)

    model.load_state_dict(torch.load('Model_2_1_1_1'))
    model.eval()

    _train_loaders, _valid_loaders = get_mamico_loaders(file_names=2)
    for _valid_loader in _valid_loaders:
        resetPipeline(model)
        _results = valid_hybrid(
            loader=_valid_loader,
            model=model,
            criterion=nn.L1Loss(),
            scaler=torch.cuda.amp.GradScaler()
        )

    pass


if __name__ == "__main__":

    main()
