import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import UNET
# MSLELoss, check_accuracy, save3DArray2File
from utils import get_loaders, get_5_loaders, get_loaders_test
from drawing_board import save3D_RGBArray2File

plt.style.use(['science'])

# Hyperparameters etc.
FEATURES = [4]
TIMESTEPS = 1000
COUETTE_DIM = 31
SIGMA = 0.3
LEARNING_RATE = 2e-3
# LOSSFN = nn.L1Loss()
# LOSS_FN = 'MAE'
LOSSFN = nn.MSELoss()
LOSS_FN = 'MSE'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 20             # 30
NUM_WORKERS = 4             # guideline: 4* num_GPU
IMAGE_HEIGHT = 128          # 1280 originally
IMAGE_WIDTH = 128           # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # The train function will complete one epoch of the training cycle.
    loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
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

        # Next consider the backward training path, especially the corresponding
        # scaler which is an object of the class GRADIENT SCALING:
        #
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
        optimizer.zero_grad()
        # .zero_grad(): Sets the gradients of all optimized torch.Tensors to 0.
        #
        scaler.scale(loss).backward()
        # .scale(): Multiplies (‘scales’) a tensor or list of tensors by the
        # scale factor and returns scaled outputs. If this instance of
        # GradScaler is not enabled, outputs are returned unmodified.
        #
        # .backward(): Computes the gradient of current tensor w.r.t. graph
        # leaves. This function accumulates gradients in the leaves - you might
        # need to zero .grad attributes or set them to None before calling it.
        #
        scaler.step(optimizer)
        # .step(): gradients automatically unscaled and returns the return
        # value of optimizer.step()
        #
        scaler.update()
        # .update():
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        # postfix(): Specify additional stats to display at the end of the bar.

    return loss


def val_fn(loader, model, loss_fn, trial_string, loss_string):

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            # torch.save(predictions, 'predictions.txt')
            # torch.save(targets, 'targets.txt')
            predict_array = predictions.cpu().detach().numpy()
            # print(f'Predict_array datatype: {type(predict_array)}')
            target_array = targets.cpu().detach().numpy()
            # print(f'Target_array datatype: {type(target_array)}')
            save3D_RGBArray2File(
                predict_array, f'predictions_{trial_string}_{loss_string}')
            save3D_RGBArray2File(
                target_array, f'targets_{trial_string}_{loss_string}')
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
            predictions = model(data)
            predict_array = predictions.cpu().detach().numpy()
            target_array = targets.cpu().detach().numpy()
            save3D_RGBArray2File(predict_array, f'T_{i}_pred_{LOSS_FN_}')
            save3D_RGBArray2File(target_array, f'T_{i}_target_{LOSS_FN_}')
            loss = loss_fn(predictions.float(), targets.float())

        loop.set_postfix(loss=loss.item())

    return loss


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

        losses.append(val_fn(valid_loader, model, loss_fn, '1', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')


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

        for epoch in range(e):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)

        losses.append(val_fn(valid_loader, model, loss_fn, '2', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')


def trial_3():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 3           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8, 16, 32]                                      # List of features
    a = [0.001, 0.002]                                  # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 40                                              # Number of epochs

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

            losses.append(val_fn(valid_loader, model,
                          loss_fn, f'3_{l+1}e-3', loss[2*i+1]))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(
                f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(' ')
            print(' ')


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

            losses.append(val_fn(valid_loader, model, loss_fn,
                          f'4_epoch_{e[j]}', loss[2*i+1]))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(
                f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(' ')
            print(' ')


def trial_5():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 5           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4, 8]                                      # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 20                                              # Number of epochs

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

        losses.append(val_fn(valid_loader, model, loss_fn, '5', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')


def trial_6():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@            TRIAL 6           @@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    t = 1000                                            # Timesteps
    d = 31                                              # Vertical resolution
    s = 0.3                                             # Sigma
    acti = 'ReLU'                                       # Activation function
    loss = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']    # Loss function
    f = [4]                                      # List of features
    a = 0.002                                           # Alpha (learning rate)
    b = 32                                              # Batch size
    e = 20                                              # Number of epochs

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

        losses.append(val_fn(valid_loader, model, loss_fn, '6', loss[2*i+1]))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'@@@@@@@@@@ T-Error:{losses[-2]:.3f}            V-Error:{losses[-1]:.3f} @@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(' ')
        print(' ')


def main2():
    l_functions = [nn.L1Loss(), 'MAE', nn.MSELoss(), 'MSE']

    for i in range(2):
        LOSSFN_ = l_functions[2*i]
        LOSS_FN_ = l_functions[2*i+1]
        displayHyperparameters(LOSS_FN_)

        # Instantiate model, define loss function, optimizer and other utils.
        model = UNET(in_channels=3, out_channels=3,
                     features=FEATURES).to(DEVICE)
        loss_fn = LOSSFN_
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_loader, test_1_loader, test_2_loader, test_3_loader, test_4_loader = get_loaders_test(
            BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TIMESTEPS, COUETTE_DIM, SIGMA)
        test_loaders = [test_1_loader, test_2_loader,
                        test_3_loader, test_4_loader]
        scaler = torch.cuda.amp.GradScaler()
        training_loss = 0.0
        losses = []

        for epoch in range(NUM_EPOCHS):
            training_loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler)
            losses.append(training_loss)

        print(
            f'The model currently yields a training loss of: {training_loss}.')

        for i in range(0, 4):
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            val_loss = test_fn(
                test_loaders[i], model, loss_fn, LOSS_FN_, (i+1))
            print(
                f'Test 0{i+1}: The model currently yields a loss of: {val_loss}.')


def main():
    trial_4()


if __name__ == "__main__":
    main()
