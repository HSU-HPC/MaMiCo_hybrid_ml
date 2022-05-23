import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import UNET, INTERIM_MD_UNET, RNN, GRU, LSTM
import time
# MSLELoss, check_accuracy, save3DArray2File
from utils import get_loaders, get_5_loaders, get_loaders_test, losses2file, get_loaders_from_file, get_loaders_from_file2
from drawing_board import save3D_RGBArray2File

plt.style.use(['science'])
np.set_printoptions(precision=6)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4             # guideline: 4* num_GPU
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # The train function will complete one epoch of the training cycle.
    loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).
    epoch_loss = 0
    counter = 0
    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        # print("Checking dimension of input data: ", data.shape)
        targets = targets.float().to(device=DEVICE)
        # print("Checking dimension of target data: ", targets.shape)
        # data = data.float().unsqueeze(1).to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # First consider the forward training path. This means calculate the
        # the predictions and determine the resultung error using the loss_fn.
        with torch.cuda.amp.autocast():
            # torch.cuda.amp and torch provide convenience methods for mixed
            # precision, where some operations use the torch.float32 (float)
            # datatype and other operations use torch.float16 (half). Some ops,
            # like linear layers and convolutions, are much faster in float16.
            # Other ops, like reductions, often require the dynamic range of
            # float32. Mixed precision tries to match each op to its appropriate
            # datatype.
            predictions = model(data)
            loss = loss_fn(predictions.float(), targets.float())
            print(loss.item())
            epoch_loss += loss.item()
            counter += 1

        # Next consider the backward training path, especially the corresponding
        # scaler which is an object of the class GRADIENT SCALING:
        #
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # If the forward pass for a particular op has float16 inputs, the
        # backward pass for that op will produce float16 gradients. Gradient
        # values with small magnitudes may not be representable in float16.
        # These values will flush to zero (“underflow”), so the update for the
        # corresponding parameters will be lost.
        #
        # To prevent underflow, “gradient scaling” multiplies the network’s
        # loss(es) by a scale factor and invokes a backward pass on the scaled
        # loss(es). Gradients flowing backward through the network are then
        # scaled by the same factor. In other words, gradient values have a
        # larger magnitude, so they don’t flush to zero.
        #
        # Each parameter’s gradient (.grad attribute) should be unscaled before
        # the optimizer updates the parameters, so the scale factor does not
        # interfere with the learning rate.
        # ---> optimizer.zero_grad()
        # .zero_grad(): Sets the gradients of all optimized torch.Tensors to 0.
        #
        # ---> scaler.scale(loss).backward()
        # .scale(): Multiplies (‘scales’) a tensor or list of tensors by the
        # scale factor and returns scaled outputs. If this instance of
        # GradScaler is not enabled, outputs are returned unmodified.
        #
        # .backward(): Computes the gradient of current tensor w.r.t. graph
        # leaves. This function accumulates gradients in the leaves - you might
        # need to zero .grad attributes or set them to None before calling it.
        #
        # ---> scaler.step(optimizer)
        # .step(): gradients automatically unscaled and returns the return
        # value of optimizer.step()
        #
        # ---> scaler.update()
        # .update():
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        # postfix(): Specify additional stats to display at the end of the bar.

    return epoch_loss/counter


def train_lstm(loader, model, optimizer, criterion, scaler):
    losses = []
    counter = 0
    time_buffer = 0
    max_loss = 0
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        # print("Checking dimension of input  data: ", data.shape)
        data = data.float().squeeze(1).to(device)
        targets = targets.float().to(device)
        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            # print("Checking dimension of output data: ", scores.shape)
            # print("Checking dimension of target data: ", targets.shape)
            loss = criterion(scores, targets)
            losses.append(loss.item())
            # print(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
        counter += 1
        if loss > max_loss:
            max_loss = loss
            time_buffer = counter
    # print('Length of losses list in train_LSTM(): ', len(losses))
    max_loss = max(losses)
    min_loss = min(losses)
    final_loss = losses[-1]
    average_loss = sum(losses)/len(losses)
    print('Losses for the first 10 inputs:')
    print(f'[0]: {losses[0]:.7f}, [1]: {losses[1]:.7f}, [2]: {losses[2]:.7f}, [3]: {losses[3]:.7f}, [4]: {losses[4]:.7f},')
    print(f'[5]: {losses[5]:.7f}, [6]: {losses[6]:.7f}, [7]: {losses[7]:.7f}, [8]: {losses[8]:.7f}, [9]: {losses[9]:.7f},')
    print(f'Max loss at t={time_buffer}: {max_loss:.7f}, Min loss: {min_loss:.7f}, Final loss: {final_loss:.7f}, Average loss: {average_loss:.7f}.')

    return [max_loss, min_loss, final_loss, average_loss]


def val_fn(loader, model, loss_fn, trial_string, loss_string):

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions, _ = model(data)
            # torch.save(predictions, 'predictions.txt')
            # torch.save(targets, 'targets.txt')
            predict_array = predictions.cpu().detach().numpy()
            # print(f'Predict_array datatype: {type(predict_array)}')
            target_array = targets.cpu().detach().numpy()
            # print(f'Target_array datatype: {type(target_array)}')
            # save3D_RGBArray2File(predict_array, f'predictions_{trial_string}_{loss_string}')
            # save3D_RGBArray2File(target_array, f'targets_{trial_string}_{loss_string}')
            # print(f'Prediction datatype: {type(predictions)}')
            # print(f'Prediction shape: {predictions.shape}')
            loss = loss_fn(predictions.float(), targets.float())

        loop.set_postfix(loss=loss.item())

    return loss


def test_fn(loader, model, loss_fn, LOSS_FN_, i):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions, _ = model(data)
            # predict_array = predictions.cpu().detach().numpy()
            # target_array = targets.cpu().detach().numpy()
            # save3D_RGBArray2File(predict_array, f'T_{i}_pred_{LOSS_FN_}')
            # save3D_RGBArray2File(target_array, f'T_{i}_target_{LOSS_FN_}')
            loss = loss_fn(predictions.float(), targets.float())

        loop.set_postfix(loss=loss.item())

    return loss


def get_latent_spaces(loader, model, loss_fn):
    loop = tqdm(loader)
    latent_spaces = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions, latent_space = model(data)
            latent_space = latent_space.cpu().detach().numpy()
            # target_array = targets.cpu().detach().numpy()
            # save3D_RGBArray2File(predict_array, f'T_{i}_pred_{LOSS_FN_}')
            # save3D_RGBArray2File(target_array, f'T_{i}_target_{LOSS_FN_}')
            loss = loss_fn(predictions.float(), targets.float())
            latent_spaces.append(latent_space)

        loop.set_postfix(loss=loss.item())

    latent_spaces_np = np.zeros((1, 64, 2, 2, 2))
    for element in latent_spaces:
        latent_spaces_np = np.vstack((latent_spaces_np, element))

    latent_spaces_np = latent_spaces_np[1:, :, :, :, :]

    return latent_spaces_np


def displayHyperparameters(timesteps_, couette_dim_, sigma_, loss_fn_, activation_, features_, learning_rate_, batch_size_, num_epochs_):
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Currently using device (cuda/CPU): {DEVICE}.')
    print('Current Trial Parameters and Model Hyperparameters:')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Couette timesteps: {timesteps_}')
    print(
        f'Spatial Resolution: {couette_dim_+1} x {couette_dim_+1} x {couette_dim_+1}')
    print(f'Noise level: {sigma_*100}% of U_wall')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Loss function: {loss_fn_}')
    print(f'Activation function: {activation_}')
    print(f'Model depth as dictated by len(features): {len(features_)}')
    print(f'Learning rate: {learning_rate_}.')
    print(f'Batch size: {batch_size_}')
    print(f'Number of epochs: {num_epochs_}.')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


def trial_1():

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 1           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16]                                      # List of features
    a = 0.001                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 40                                              # Number of epochs
    key_list = ['1_MAE_ReLU_Train_Error', '1_MAE_ReLU_Valid_Error',
                '1_MSE_ReLU_Train_Error', '1_MSE_ReLU_Valid_Error']
    results_dict = {}
    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=f).to(DEVICE)
        loss_fn = loss[2*i]
        optimizer = optim.Adam(model.parameters(), lr=a)
        train_loader, valid_loader = get_loaders(
            b, NUM_WORKERS, PIN_MEMORY, t, d, s)

        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []
        start = time.time()
        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss.item())
        end = time.time()
        losses2file(losses, f'trial_1_{loss[2*i+1]}')
        print(f'@@@@@@@@@@ Duration:{end-start} @@@@@@@@@@')
        losses.append(val_fn(valid_loader, model, loss_fn, '1', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')
        errors = {key_list[2*i]: losses[-2], key_list[2*i+1]: losses[-1]}
        results_dict.update(errors)

    return results_dict


def trial_2():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 2           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'Tanh'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16]                                      # List of features
    a = 0.001                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 40                                              # Number of epochs
    key_list = ['2_MAE_Tanh_Train_Error', '2_MAE_Tanh_Valid_Error',
                '2_MSE_Tanh_Train_Error', '2_MSE_Tanh_Valid_Error']
    results_dict = {}
    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=f, activation=nn.Tanh()).to(DEVICE)
        loss_fn = loss[2*i]
        optimizer = optim.Adam(model.parameters(), lr=a)
        train_loader, valid_loader = get_loaders(
            b, NUM_WORKERS, PIN_MEMORY, t, d, s)

        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []
        start = time.time()
        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)
        end = time.time()
        losses2file(losses, f'trial_2_{loss[2*i+1]}')
        print(f'@@@@@@@@@@ Duration:{end-start} @@@@@@@@@@')
        losses.append(val_fn(valid_loader, model, loss_fn, '2', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')
        errors = {key_list[2*i]: losses[-2], key_list[2*i+1]: losses[-1]}
        results_dict.update(errors)

    return results_dict


def trial_3():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 3           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16, 32]                                  # List of features
    a = [0.001, 0.002]                                  # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 40                                              # Number of epochs
    key_list = ['3_MAE_alpha_1e-3_Train_Error', '3_MAE_alpha_1e-3_Valid_Error',
                '3_MAE_alpha_2e-3_Train_Error', '3_MAE_alpha_2e-3_Valid_Error',
                '3_MSE_alpha_1e-3_Train_Error', '3_MSE_alpha_1e-3_Valid_Error',
                '3_MSE_alpha_2e-3_Train_Error', '3_MSE_alpha_2e-3_Valid_Error']
    results_dict = {}
    c = 0
    for i in range(2):
        for l in range(2):
            displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a[l], b, e)

            # Instantiate model, define loss function, optimizer and other utils.
            model = UNET(in_channels=3, out_channels=3,
                         features=f).to(DEVICE)
            loss_fn = loss[2*i]
            optimizer = optim.Adam(model.parameters(), lr=a[l])
            train_loader, valid_loader = get_loaders(
                b, NUM_WORKERS, PIN_MEMORY, t, d, s)

            scaler = torch.cuda.amp.GradScaler()
            training_loss = 0.0
            losses = []

            for epoch in range(e):
                training_loss = train_fn(
                    train_loader, model, optimizer, loss_fn, scaler)
                losses.append(training_loss)

            losses2file(losses, f'trial_3_{loss[2*i+1]}_{l+1}e-3')

            losses.append(val_fn(valid_loader, model,
                          loss_fn, f'3_{l+1}e-3', loss[2*i+1]))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(
                f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(' ')
            print(' ')
            errors = {key_list[2*c]: losses[-2], key_list[2*c+1]: losses[-1]}
            results_dict.update(errors)
            c += 1
    return results_dict


def trial_4():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 4           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16]                                      # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    # Number of epochs
    e = [40, 20]
    key_list = ['4_MAE_epochs_40_Train_Error', '4_MAE_epochs_40_Valid_Error',
                '4_MAE_epochs_20_Train_Error', '4_MAE_epochs_20_Valid_Error',
                '4_MSE_epochs_40_Train_Error', '4_MSE_epochs_40_Valid_Error',
                '4_MSE_epochs_20_Train_Error', '4_MSE_epochs_20_Valid_Error']
    results_dict = {}
    c = 0
    for i in range(2):
        for j in range(2):
            displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e[j])

            # Instantiate model, define loss function, optimizer and other utils.
            model = UNET(in_channels=3, out_channels=3,
                         features=f).to(DEVICE)
            loss_fn = loss[2*i]
            optimizer = optim.Adam(model.parameters(), lr=a)
            train_loader, valid_loader = get_5_loaders(
                b, NUM_WORKERS, PIN_MEMORY, t, d, s)

            scaler = torch.cuda.amp.GradScaler()
            training_loss = 0.0
            losses = []

            for epoch in range(e[j]):
                training_loss = train_fn(
                    train_loader, model, optimizer, loss_fn, scaler)
                losses.append(training_loss)

            losses2file(losses, f'trial_4_{loss[2*i+1]}_{str(e[j])}')

            losses.append(val_fn(valid_loader, model, loss_fn,
                          f'4_epoch_{e[j]}', loss[2*i+1]))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(
                f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(' ')
            print(' ')
            errors = {key_list[2*c]: losses[-2], key_list[2*c+1]: losses[-1]}
            results_dict.update(errors)
            c += 1
    return results_dict


def trial_5():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 5           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8]                                          # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 20                                              # Number of epochs
    key_list = ['5_MAE_4_8_Train_Error', '5_MAE_4_8_Valid_Error',
                '5_MSE_4_8_Train_Error', '5_MSE_4_8_Valid_Error']
    results_dict = {}
    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=f).to(DEVICE)
        loss_fn = loss[2*i]
        optimizer = optim.Adam(model.parameters(), lr=a)
        train_loader, valid_loader = get_loaders(
            b, NUM_WORKERS, PIN_MEMORY, t, d, s)

        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []

        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)

        losses2file(losses, f'trial_5_{loss[2*i+1]}')

        losses.append(val_fn(valid_loader, model, loss_fn, '5', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')
        errors = {key_list[2*i]: losses[-2], key_list[2*i+1]: losses[-1]}
        results_dict.update(errors)

    return results_dict


def trial_6():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 6           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4]                                             # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 20                                              # Number of epochs
    key_list = ['6_MAE_4_Train_Error', '6_MAE_4_Valid_Error',
                '6_MSE_4_Train_Error', '6_MSE_4_Valid_Error']
    results_dict = {}
    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=f).to(DEVICE)
        loss_fn = loss[2*i]
        optimizer = optim.Adam(model.parameters(), lr=a)
        train_loader, valid_loader = get_loaders(
            b, NUM_WORKERS, PIN_MEMORY, t, d, s)

        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []

        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)

        losses2file(losses, f'trial_6_{loss[2*i+1]}')

        losses.append(val_fn(valid_loader, model, loss_fn, '6', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')
        errors = {key_list[2*i]: losses[-2], key_list[2*i+1]: losses[-1]}
        results_dict.update(errors)

    return results_dict


def trial_7():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 7           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16, 32]                                   # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 15                                              # Number of epochs

    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model and other utils.
        model = INTERIM_MD_UNET(
            in_channels=3, out_channels=3, features=f).to(DEVICE)
        # Define loss function
        loss_fn = loss[2*i]
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=a)
        # Instantiate other utils
        train_loader, valid_loader = get_loaders(
            b, NUM_WORKERS, PIN_MEMORY, t, d, s)
        # Prepare training cycle
        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0

        # Training cycle
        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)

        # Latent spaces via validation set
        latent_spaces = get_latent_spaces(valid_loader, model, loss_fn)
        print(latent_spaces.shape)
        save3D_RGBArray2File(latent_spaces, f"latent_space_test_{loss[2*i+1]}")


def trial_8():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@         TRIAL 8 LSTM         @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # Alpha (learning rate)
    a = 0.0005
    b = 8                                               # Batch size
    e = 25                                               # Number of epochs

    model = LSTM(
        input_size=512, hidden_size=1024, num_layers=2, device=device).to(device)
    # Define loss function
    loss_fn = nn.MSELoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=a)
    # Instantiate other utils
    train_loader = get_loaders_from_file(
        batch_size=b, num_workers=4, pin_memory=True)
    # Prepare training cycle
    scaler = torch.cuda.amp.GradScaler()
    training_loss = 0.0
    names = ['RNN', 'GRU', 'LSTM']
    num_layers = [2, 4, 8]
    learning_rates = [0.0005, 0.0001, 0.00005]
    max_losses = []
    min_losses = []
    final_losses = []
    average_losses = []

    # Training cycle
    for epoch in range(e):
        training_loss = train_lstm(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=loss_fn,
            scaler=scaler
        )
        max_losses.append(training_loss[0])
        min_losses.append(training_loss[1])
        final_losses.append(training_loss[2])
        average_losses.append(training_loss[-1])

    # Print model summary
    model_summary(
        name=names[2],
        num_layers=num_layers[0],
        learning_rate=learning_rates[0],
        epochs=e,
        max_losses=max_losses,
        min_losses=min_losses,
        final_losses=final_losses,
        average_losses=average_losses
    )


def first_trial_RNNs():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@      FIRST TRIAL RNNs        @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    batch = 1                                                # Batch size
    epoch = 50                                               # Number of epochs
    names = ['RNN', 'GRU', 'LSTM']
    num_layers = [2, 4, 8]
    learning_rates = [0.0005, 0.0001, 0.00005]

    # Loop for RNN
    for i in range(1):                                      # num_layers
        for j in range(2, 3):                                  # learning_rates
            # First, instantiate the ML model
            model = RNN(
                input_size=512,
                hidden_size=1024,
                num_layers=num_layers[i],
                device=device
            ).to(device)
            # Second, define loss function and optimizer
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rates[j])
            # Third, instantiate loader
            train_loader = get_loaders_from_file(
                batch_size=batch,
                num_workers=4,
                pin_memory=True
            )
            # Fourth, instantiate remaining utils: scaler and loss containers
            scaler = torch.cuda.amp.GradScaler()
            max_losses = []
            min_losses = []
            final_losses = []
            average_losses = []
            # Fifth, loop through the epochs and perform training
            for e in range(epoch):
                training_loss = train_lstm(
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    scaler=scaler
                )
                max_losses.append(training_loss[0])
                min_losses.append(training_loss[1])
                final_losses.append(training_loss[2])
                average_losses.append(training_loss[3])
            # Finally, print out model summary
            model_summary(
                name=names[0],
                num_layers=num_layers[i],
                learning_rate=learning_rates[j],
                epochs=epoch,
                max_losses=max_losses,
                min_losses=min_losses,
                final_losses=final_losses,
                average_losses=average_losses
            )

    # Repeat loop for GRU
    for i in range(3):                                      # num_layers
        for j in range(3):                                  # learning_rates
            # First, instantiate the ML model
            model = GRU(
                input_size=512,
                hidden_size=1024,
                num_layers=num_layers[i],
                device=device
            ).to(device)
            # Second, define loss function and optimizer
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rates[j])
            # Third, instantiate loader
            train_loader = get_loaders_from_file(
                batch_size=batch,
                num_workers=4,
                pin_memory=True
            )
            # Fourth, instantiate remaining utils: scaler and loss containers
            scaler = torch.cuda.amp.GradScaler()
            max_losses = []
            min_losses = []
            final_losses = []
            average_losses = []
            # Fifth, loop through the epochs and perform training
            for e in range(epoch):
                training_loss = train_lstm(
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    scaler=scaler
                )
                max_losses.append(training_loss[0])
                min_losses.append(training_loss[1])
                final_losses.append(training_loss[2])
                average_losses.append(training_loss[3])
            # Finally, print out model summary
            model_summary(
                name=names[1],
                num_layers=num_layers[i],
                learning_rate=learning_rates[j],
                epochs=epoch,
                max_losses=max_losses,
                min_losses=min_losses,
                final_losses=final_losses,
                average_losses=average_losses
            )

    # Repeat loop for LSTM
    for i in range(3):                                      # num_layers
        for j in range(3):                                  # learning_rates
            # First, instantiate the ML model
            model = LSTM(
                input_size=512,
                hidden_size=1024,
                num_layers=num_layers[i],
                device=device
            ).to(device)
            # Second, define loss function and optimizer
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rates[j])
            # Third, instantiate loader
            train_loader = get_loaders_from_file(
                batch_size=batch,
                num_workers=4,
                pin_memory=True
            )
            # Fourth, instantiate remaining utils: scaler and loss containers
            scaler = torch.cuda.amp.GradScaler()
            max_losses = []
            min_losses = []
            final_losses = []
            average_losses = []
            # Fifth, loop through the epochs and perform training
            for e in range(epoch):
                training_loss = train_lstm(
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    scaler=scaler
                )
                max_losses.append(training_loss[0])
                min_losses.append(training_loss[1])
                final_losses.append(training_loss[2])
                average_losses.append(training_loss[3])
            # Finally, print out model summary
            model_summary(
                name=names[2],
                num_layers=num_layers[i],
                learning_rate=learning_rates[j],
                epochs=epoch,
                max_losses=max_losses,
                min_losses=min_losses,
                final_losses=final_losses,
                average_losses=average_losses
            )


def second_trial_RNNs():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@      SECOND TRIAL RNNs       @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    batch = 8                                                # Batch size
    epoch = 100                                               # Number of epochs
    names = ['RNN', 'GRU', 'LSTM']
    num_layers = [2, 4, 8]
    learning_rates = [0.00005, 0.00001, 0.000005]

    # Loop for RNN
    for j in range(3):                                  # learning_rates
        # First, instantiate the ML model
        model = RNN(
            input_size=512,
            hidden_size=1024,
            num_layers=num_layers[0],
            device=device
        ).to(device)
        # Second, define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[j])
        # Third, instantiate loader
        train_loader = get_loaders_from_file(
            batch_size=batch,
            num_workers=4,
            pin_memory=True
        )
        # Fourth, instantiate remaining utils: scaler and loss containers
        scaler = torch.cuda.amp.GradScaler()
        max_losses = []
        min_losses = []
        final_losses = []
        average_losses = []
        # Fifth, loop through the epochs and perform training
        for e in range(epoch):
            training_loss = train_lstm(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                criterion=loss_fn,
                scaler=scaler
            )
            max_losses.append(training_loss[0])
            min_losses.append(training_loss[1])
            final_losses.append(training_loss[2])
            average_losses.append(training_loss[3])
        # Finally, print out model summary
        model_summary(
            name=names[0],
            num_layers=num_layers[0],
            learning_rate=learning_rates[j],
            epochs=epoch,
            max_losses=max_losses,
            min_losses=min_losses,
            final_losses=final_losses,
            average_losses=average_losses
        )

    # Loop for GRU
    for j in range(3):                                  # learning_rates
        # First, instantiate the ML model
        model = GRU(
            input_size=512,
            hidden_size=1024,
            num_layers=num_layers[0],
            device=device
        ).to(device)
        # Second, define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[j])
        # Third, instantiate loader
        train_loader = get_loaders_from_file(
            batch_size=batch,
            num_workers=4,
            pin_memory=True
        )
        # Fourth, instantiate remaining utils: scaler and loss containers
        scaler = torch.cuda.amp.GradScaler()
        max_losses = []
        min_losses = []
        final_losses = []
        average_losses = []
        # Fifth, loop through the epochs and perform training
        for e in range(epoch):
            training_loss = train_lstm(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                criterion=loss_fn,
                scaler=scaler
            )
            max_losses.append(training_loss[0])
            min_losses.append(training_loss[1])
            final_losses.append(training_loss[2])
            average_losses.append(training_loss[3])
        # Finally, print out model summary
        model_summary(
            name=names[1],
            num_layers=num_layers[0],
            learning_rate=learning_rates[j],
            epochs=epoch,
            max_losses=max_losses,
            min_losses=min_losses,
            final_losses=final_losses,
            average_losses=average_losses
        )

    # Loop for LSTM
    for j in range(3):                                  # learning_rates
        # First, instantiate the ML model
        model = LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=num_layers[0],
            device=device
        ).to(device)
        # Second, define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[j])
        # Third, instantiate loader
        train_loader = get_loaders_from_file(
            batch_size=batch,
            num_workers=4,
            pin_memory=True
        )
        # Fourth, instantiate remaining utils: scaler and loss containers
        scaler = torch.cuda.amp.GradScaler()
        max_losses = []
        min_losses = []
        final_losses = []
        average_losses = []
        # Fifth, loop through the epochs and perform training
        for e in range(epoch):
            training_loss = train_lstm(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                criterion=loss_fn,
                scaler=scaler
            )
            max_losses.append(training_loss[0])
            min_losses.append(training_loss[1])
            final_losses.append(training_loss[2])
            average_losses.append(training_loss[3])
        # Finally, print out model summary
        model_summary(
            name=names[2],
            num_layers=num_layers[0],
            learning_rate=learning_rates[j],
            epochs=epoch,
            max_losses=max_losses,
            min_losses=min_losses,
            final_losses=final_losses,
            average_losses=average_losses
        )


def third_trial_RNN():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@       Third TRIAL RNNs       @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    batch = 1                                                # Batch size
    epoch = 100                                               # Number of epochs
    names = ['RNN', 'GRU', 'LSTM']
    num_layers = [2, 4, 8]
    learning_rates = [0.00005, 0.00001, 0.000005]

    # Loop for RNN
    for j in range(2, 3):                                  # learning_rates
        # First, instantiate the ML model
        model = RNN(
            input_size=512,
            hidden_size=1024,
            num_layers=num_layers[0],
            device=device
        ).to(device)
        # Second, define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[j])
        # Third, instantiate loader
        train_loader = get_loaders_from_file2(
            batch_size=batch,
            num_workers=4,
            pin_memory=True
        )
        # Fourth, instantiate remaining utils: scaler and loss containers
        scaler = torch.cuda.amp.GradScaler()
        max_losses = []
        min_losses = []
        final_losses = []
        average_losses = []
        # Fifth, loop through the epochs and perform training
        for e in range(epoch):
            training_loss = train_lstm(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                criterion=loss_fn,
                scaler=scaler
            )
            max_losses.append(training_loss[0])
            min_losses.append(training_loss[1])
            final_losses.append(training_loss[2])
            average_losses.append(training_loss[3])
        # Finally, print out model summary
        model_summary(
            name=names[0],
            num_layers=num_layers[0],
            learning_rate=learning_rates[j],
            epochs=epoch,
            max_losses=max_losses,
            min_losses=min_losses,
            final_losses=final_losses,
            average_losses=average_losses
        )


def model_summary(name, num_layers, learning_rate, epochs, max_losses, min_losses, final_losses, average_losses):
    print("@@@@@@@@@@@@@@@         MODEL SUMMARY         @@@@@@@@@@@@@@@")
    print(f'Name: {name}')
    print(f'Num layers: {num_layers}')
    print(f'Learning rate: {learning_rate}')
    print(f'Num epochs: {epochs}')

    for i in range(epochs):
        if i < 9:
            print(
                f'Epoch: 0{i+1}, Max loss: {max_losses[i]:.7f}, Min loss: {min_losses[i]:.7f}, Final loss: {final_losses[i]:.7f}, Average loss: {average_losses[i]:.7f}.')
        else:
            print(
                f'Epoch: {i+1}, Max loss: {max_losses[i]:.7f}, Min loss: {min_losses[i]:.7f}, Final loss: {final_losses[i]:.7f}, Average loss: {average_losses[i]:.7f}.')


def first_trial_hybrid():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@      FIRST TRIAL Hybrid      @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.MSELoss(), 'MSE']                        # Loss function
    f = [4, 8, 16, 32]                                  # List of features
    a = [0.0001, 0.00005]                               # Alpha (learning rate)
    b = 1                                               # Batch size
    e = 150                                             # Number of epochs

    key_list = ['H1_MSE_alpha_1e-3_Train_Error', 'H1_MSE_alpha_5e-4_Valid_Error',
                'H1_MSE_alpha_1e-3_Train_Error', 'H1_MSE_alpha_5e-4_Valid_Error']
    results_dict = {}
    # Create counter to track
    c = 0
    for i in range(1):                                  # Index for loss function
        for j in range(2):                              # Index for learning rates
            displayHyperparameters(t, d, s, loss[1], acti, f, a[j], b, e)

            # Instantiate model
            model = INTERIM_MD_UNET(
                device=device,
                in_channels=3,
                out_channels=3,
                features=[4, 8, 16, 32],
                activation=nn.ReLU(inplace=True),
                RNN_in_size=512,
                RNN_hid_size=1024,
                RNN_lay=2
            ).to(device)

            # Define loss function and optimizer
            loss_fn = loss[0]
            optimizer = optim.Adam(model.parameters(), lr=a[j])

            # Create train and valid loaders
            train_loader, valid_loader = get_loaders(
                b, NUM_WORKERS, PIN_MEMORY, t, d, s)

            # Define other utils: scaler, loss placeholder, placeholder container
            scaler = torch.cuda.amp.GradScaler()
            training_loss = 0.0
            losses = []

            # Initiate training loop and append average epoch loss to container
            for epoch in range(e):
                training_loss = train_fn(
                    train_loader, model, optimizer, loss_fn, scaler)
                losses.append(training_loss)

            # Save losses to file for later visualization of training progress
            losses2file(losses, f'trial_3_{loss[2*i+1]}_{j+1}e-3')

            # Perform validation set to check proof of concept
            losses.append(val_fn(valid_loader, model, loss_fn,
                          f'3_{j+1}e-3', loss[2*i+1]))

            # Print statements for quick feedback
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(
                f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(' ')
            print(' ')

            # create dictionary to hold training and validation errors
            errors = {key_list[2*c]: losses[-2], key_list[2*c+1]: losses[-1]}
            results_dict.update(errors)
            c += 1
    return results_dict


def tests():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            MODEL 1           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    # List of features
    f = [4, 8, 16]
    # Alpha (learning rate)
    a = 0.002
    b = 32                                              # Batch size
    e = 20
    key_list = ['T_1_MAE_Test_Error', 'T_2_MAE_Test_Error',
                'T_3_MAE_Test_Error', 'T_4_MAE_Test_Error',
                'T_1_MSE_Test_Error', 'T_2_MSE_Test_Error',
                'T_3_MSE_Test_Error', 'T_4_MSE_Test_Error', ]
    results_dict = {}
    c = 0

    for i in range(2):
        displayHyperparameters(t, d, s, loss[2*i+1], acti, f, a, b, e)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=f).to(DEVICE)
        loss_fn = loss[2*i]
        optimizer = optim.Adam(model.parameters(), lr=a)
        train_loader, valid_loader = get_loaders(
                b, NUM_WORKERS, PIN_MEMORY, t, d, s)

        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []
        epoch = 0

        while epoch < e:
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)
            epoch += 1

        if epoch == e:
            test_loader_1, test_loader_2, test_loader_3, test_loader_4 = get_loaders_test(
                b, NUM_WORKERS, PIN_MEMORY)
            test_loaders = [test_loader_1, test_loader_2,
                            test_loader_3, test_loader_4]

            for j in range(0, 4):
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print(
                    f'@@@@@@@@@@@@@@@          TEST {j+1} {loss[2*i+1]}         @@@@@@@@@@@@@@@')
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                val_loss = test_fn(
                    test_loaders[j], model, loss_fn, loss[2*i+1], (j+1))
                print(
                    f'Test 0{j+1}: The model currently yields a loss of: {val_loss}.')
                errors = {key_list[c]: val_loss}
                results_dict.update(errors)
                c += 1
    return results_dict


def main():
    first_trial_RNNs()
    third_trial_RNN()
    # dict = first_trial_hybrid()
    # for key, value in dict.items():
    #     print('{} : {}'.format(key, value))

    '''
    x = torch.zeros((5, 64, 2, 2, 2))
    print(x.shape)
    print(x[0, 0, 0, 0, 0])
    x = x[1:]
    print(x.shape)
    print(x[0, 0, 0, 0, 0])
    x = torch.vstack((x, torch.rand(1, 64, 2, 2, 2)))
    print(x.shape)
    print(x[0, 0, 0, 0, 0])
    print(x[-1, -1, -1, -1, -1])


    dict = {}
    key_list = ['3_MAE_1e-3_Train_Error', '3_MAE_1e-3_Valid_Error',
                '3_MAE_2e-3_Train_Error', '3_MAE_2e-3_Valid_Error',
                '3_MSE_1e-3_Train_Error', '3_MSE_1e-3_Valid_Error',
                '3_MSE_2e-3_Train_Error', '3_MSE_2e-3_Valid_Error']
    c = 0
    for i in range(2):
        for j in range(2):
            errors = {key_list[2*c]: 2*c, key_list[2*c+1]: 2*c+1}
            dict.update(errors)
            c += 1


    for key, value in dict.items():
        print('{} : {}'.format(key, value))

    '''


if __name__ == "__main__":
    main()
