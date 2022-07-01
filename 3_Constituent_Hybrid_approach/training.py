import torch
import random
import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_UNET
from utils import get_UNET_AE_loaders, get_RNN_loaders, get_mamico_loaders, losses2file, dataset2csv, get_Hybrid_loaders
from plotting import plotAvgLoss, compareAvgLoss

torch.manual_seed(0)
random.seed(0)

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
    # print('------------------------------------------------------------')
    # print(f'{identifier} Training -> Epoch: {current_epoch}, Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
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
    # print('------------------------------------------------------------')
    # print(f'{identifier} Validation -> Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def train_HYBRID(loader, model, optimizer, criterion, scaler, alpha, current_epoch):
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


def valid_HYBRID(loader, model, criterion, scaler, alpha, current_epoch):
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


def trial_0_UNET_AE(_alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'
    _model_identifier = f'LR{_alpha_string}'
    print('Initializing UNET_AE model.')
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
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(5):
        avg_loss = 0
        for _train_loader in _train_loaders:
            avg_loss += train_AE(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                alpha=_alpha_string,
                current_epoch=epoch+1
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_AE(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                alpha=_alpha_string,
                current_epoch=0
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}')
        _epoch_valids.append(avg_valid/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_UNET_AE_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_UNET_AE_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_UNET_AE_{_model_identifier}.csv',
            f'{_file_prefix}Valids_UNET_AE_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_UNET_AE_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_UNET_AE_{_model_identifier}'
    )
    return


def trial_0_mp():
    _alphas = [0.001]  # [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_001']  # ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    # _alphas = [0.01, 0.001, 0.0001]
    # _alphas_strings = ['test1', 'test2', 'test3']
    _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=1)

    processes = []
    counter = 1

    for i in range(3):
        p = mp.Process(
            target=trial_0_UNET_AE,
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


def trial_1_RNN(_seq_length, _num_layers, _alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_RNN/'
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing RNN model.')
    _model = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = 0
        for _train_loader in _train_loaders:
            avg_loss += train_RNN(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=epoch+1
            )
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}')

        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_RNN(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=0
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}')
        _epoch_valids.append(avg_valid/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_RNN_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_RNN_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_RNN_{_model_identifier}.csv',
            f'{_file_prefix}Valids_RNN_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_RNN_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_RNN_{_model_identifier}'
    )


def trial_1_RNN_mp():
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _alphas.reverse()
    _alpha_strings.reverse()
    _rnn_depths = [1, 2, 3, 4]
    _seq_lengths = [5, 15, 25]
    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=0, sequence_length=5)
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)
    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx, _lr in enumerate(_alphas):
        for _rnn_depth in _rnn_depths:

            processes = []
            counter = 1

            for i in range(3):
                p = mp.Process(
                    target=trial_1_RNN,
                    args=(_seq_lengths[i], _rnn_depth, _lr,
                          _alpha_strings[idx], _t_loaders[i], _v_loaders[i],)
                )
                p.start()
                processes.append(p)
                print(f'Creating Process Number: {counter}')
                counter += 1

            for process in processes:
                process.join()
                print('Joining Process')


def trial_2_GRU(_seq_length, _num_layers, _alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/2_GRU/'
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing GRU model.')
    _model = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = 0
        for _train_loader in _train_loaders:
            avg_loss += train_RNN(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=epoch+1
            )
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_RNN(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=0
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}')
        _epoch_valids.append(avg_valid/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_GRU_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_GRU_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_GRU_{_model_identifier}.csv',
            f'{_file_prefix}Valids_GRU_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_GRU_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_GRU_{_model_identifier}'
    )


def trial_2_GRU_mp():
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _rnn_depths = [1, 2, 3, 4]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=0, sequence_length=5)
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx, _lr in enumerate(_alphas):
        for _rnn_depth in _rnn_depths:

            processes = []
            counter = 1

            for i in range(3):
                p = mp.Process(
                    target=trial_2_GRU,
                    args=(_seq_lengths[i], _rnn_depth, _lr,
                          _alpha_strings[idx], _t_loaders[i], _v_loaders[i],)
                )
                p.start()
                processes.append(p)
                print(f'Creating Process Number: {counter}')
                counter += 1

            for process in processes:
                process.join()
                print('Joining Process')


def trial_3_LSTM(_seq_length, _num_layers, _alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_LSTM/'
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing model.')
    _model = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_length,
        num_layers=_num_layers,
        device=device
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model.parameters(), lr=_alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = 0
        for _train_loader in _train_loaders:
            avg_loss += train_RNN(
                loader=_train_loader,
                model=_model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=epoch+1
            )
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}')

        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_RNN(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=0
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}')
        _epoch_valids.append(avg_valid/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_LSTM_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_LSTM_{_model_identifier}'
    )
    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_LSTM_{_model_identifier}.csv',
            f'{_file_prefix}Valids_LSTM_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_LSTM_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_LSTM_{_model_identifier}'
    )


def trial_3_LSTM_mp():
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _rnn_depths = [1, 2, 3, 4]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=0, sequence_length=5)
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx, _lr in enumerate(_alphas):
        for _rnn_depth in _rnn_depths:

            processes = []
            counter = 1

            for i in range(3):
                p = mp.Process(
                    target=trial_3_LSTM,
                    args=(_seq_lengths[i], _rnn_depth, _lr,
                          _alpha_strings[idx], _t_loaders[i], _v_loaders[i],)
                )
                p.start()
                processes.append(p)
                print(f'Creating Process Number: {counter}')
                counter += 1

            for process in processes:
                process.join()
                print('Joining Process')


def trial_4_Hybrid(_train_loaders, _valid_loaders):
    # _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _criterion = nn.L1Loss()
    # _train_loader, _valid_loader = get_UNET_AE_loaders(file_names=0)
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/4_Hybrid_RNN_UNET/'
    _model_identifier = 'LSTM_Seq15_Lay1_LR0_00005'
    print('Initializing model.')

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/Model_UNET_AE_0_001'))

    _model_rnn = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=15,
        num_layers=1,
        device=device
    )
    _model_rnn.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_LSTM/Model_LSTM_Seq15_Lay1_LR0_00005'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=15
    ).to(device)

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(_model_hybrid.parameters(), lr=0.001)
    _epoch_losses = []

    print('Beginning training.')
    for epoch in range(30):
        avg_loss = 0
        for _train_loader in _train_loaders:
            avg_loss += valid_RNN(
                loader=_train_loader,
                model=_model_hybrid,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=0
            )
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

    _valid_loss = 0
    for _valid_loader in _valid_loaders:
        _valid_loss += valid_RNN(
            loader=_valid_loader,
            model=_model_hybrid,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=0
        )
    print('------------------------------------------------------------')
    print(f'{_model_identifier} Validation -> Averaged Loader Loss: {_valid_loss/len(_valid_loaders):.3f}')
    _epoch_losses.append(_valid_loss/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_Hybrid_{_model_identifier}'
    )

    plotAvgLoss(
        avg_losses=_epoch_losses,
        file_prefix=_file_prefix,
        file_name=f'Hybrid_{_model_identifier}'
    )

    torch.save(
        _model_hybrid.state_dict(),
        f'{_file_prefix}Model_Hybrid_{_model_identifier}'
    )

    pass


def trial_4_Hybrid_mp():
    _train_loaders, _valid_loaders = get_Hybrid_loaders(file_names=-1)
    trial_4_Hybrid(_train_loaders, _valid_loaders)

    pass


if __name__ == "__main__":
    trial_2_GRU_mp()
    pass
