import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders   # MSLELoss, check_accuracy, save3DArray2File
from drawing_board import save3D_RGBArray2File

plt.style.use(['science'])

# Hyperparameters etc.
FEATURES = [4, 8]
TIMESTEPS = 1000
COUETTE_DIM = 31
SIGMA = 0.3
LEARNING_RATE = 2e-3
LOSSFN = nn.L1Loss()
LOSS_FN = 'nn.L1Loss()'
# LOSSFN = nn.MSELoss()
# LOSS_FN = 'nn.MSELoss()'
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


def val_fn(loader, model, loss_fn):

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
            if loss_fn == nn.L1Loss:
                save3D_RGBArray2File(predict_array, 'predictions_MAE')
                save3D_RGBArray2File(target_array, 'targets_MAE')
            else:
                save3D_RGBArray2File(predict_array, 'predictions_MSE')
                save3D_RGBArray2File(target_array, 'targets_MSE')
            # print(f'Prediction datatype: {type(predictions)}')
            # print(f'Prediction shape: {predictions.shape}')
            loss = loss_fn(predictions.float(), targets.float())

        loop.set_postfix(loss=loss.item())

    return loss


def main():

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Currently using device (cuda/CPU): {DEVICE}.')
    print('Current Trial Parameters and Model Hyperparameters:')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Couette timesteps: {TIMESTEPS}')
    print(
        f'Spatial Resolution: {COUETTE_DIM+1} x {COUETTE_DIM+1} x {COUETTE_DIM+1}')
    print(f'Noise level: {SIGMA*100}% of U_wall')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'Loss function: {LOSS_FN}')
    print('Activation function: ReLU')
    print(f'Model depth as dictated by len(features): {len(FEATURES)}')
    print(f'Learning rate: {LEARNING_RATE}.')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Number of epochs: {NUM_EPOCHS}.')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    model = UNET(in_channels=3, out_channels=3, features=FEATURES).to(DEVICE)

    loss_fn = LOSSFN
    # Defines the loss function to be MAE (=Mean Average Error).

    # loss_fn = MSLELoss()
    # Defines the loss function to be Mean Squared Logarithmic Error

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader, test_loader = get_loaders(
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TIMESTEPS, COUETTE_DIM, SIGMA)

    scaler = torch.cuda.amp.GradScaler()
    training_loss = 0.0
    losses = []

    for epoch in range(NUM_EPOCHS):
        training_loss = train_fn(
            train_loader, model, optimizer, loss_fn, scaler)
        losses.append(training_loss)
        # To save the model, refer to the original code described by Aladdin
        # Persson (YouTube, GitHub)

    # np.savetxt("losses_file.csv", losses, delimiter=", ")

    # print('Currently using validation set:')
    val_loss = val_fn(val_loader, model, loss_fn)

    # print('Currently using test set:')
    # test_loss = val_fn(test_loader, model, loss_fn)

    print(f'The model currently yields a training loss of: {training_loss}.')
    print(f'The model currently yields a val loss of: {val_loss}.')
    # print(f'The model currently yields a test loss of: {test_loss}.')


if __name__ == "__main__":
    main()
