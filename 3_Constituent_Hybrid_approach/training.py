import torch
import random
import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_UNET, resetPipeline
from utils import get_UNET_AE_loaders, get_RNN_loaders, get_mamico_loaders, losses2file, dataset2csv, get_Hybrid_loaders
from plotting import compareAvgLoss, compareLossVsValid, compareFlowProfile3x3, compareErrorTimeline_np

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 1             # guideline: 4* num_GPU
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
    # optimizer.zero_grad()

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
    # print(f'{alpha} Training -> Epoch: {current_epoch}, Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
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
    # print('------------------------------------------------------------')
    # print(f'{alpha} Validation -> Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss


def errorTimeline(loader, model, criterion):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    print('Starting errorTimeline.')
    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            losses.append(loss.item())

    return losses


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
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/Model_UNET_AE_LR0_0005'))
    _model.eval()

    _loader_1, _loader_2_ = get_UNET_AE_loaders(file_names=-2)
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/Latentspace_Dataset'
    _out_file_names = [
        '_kvs_10K_NE',
        '_kvs_10K_NW',
        '_kvs_10K_SE',
        '_kvs_10K_SW',
        '_kvs_20K_NE',
        '_kvs_20K_NW',
        '_kvs_20K_SE',
        '_kvs_20K_SW',
        '_kvs_30K_NE',
        '_kvs_30K_NW',
        '_kvs_30K_SE',
        '_kvs_30K_SW',
        '_kvs_40K_NE',
        '_kvs_40K_NW',
        '_kvs_40K_SE',
        '_kvs_40K_SW',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE(
            loader=_loader,
            model=_model,
            out_file_name=f'{_out_directory}{_out_file_names[idx]}'
        )


def train_RNN(loader, model, optimizer, criterion, scaler, identifier='', current_epoch=''):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -

    # loop = tqdm(loader)
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0
    optimizer.zero_grad()
    pred_mean = 0
    targ_mean = 0
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            pred_mean += (predictions.cpu().detach().numpy()).mean()
            targ_mean += (targets.cpu().detach().numpy()).mean()

            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # loop.set_postfix(loss=loss.item())
    avg_loss = epoch_loss/counter
    pred_mean = pred_mean/counter
    targ_mean = targ_mean/counter
    print('------------------------------------------------------------')
    print(f'{identifier} Training -> Epoch: {current_epoch}')
    print(
        f'Mean Targ LS: {targ_mean:.5f}         Mean Pred LS: {pred_mean:.5f}')
    return avg_loss


def valid_RNN(loader, model, criterion, scaler, identifier='', current_epoch=''):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    # start_time = time.time()

    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    counter = 0
    pred_mean = 0
    targ_mean = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            pred_mean += (predictions.cpu().detach().numpy()).mean()
            targ_mean += (targets.cpu().detach().numpy()).mean()
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            counter += 1

    avg_loss = epoch_loss/counter
    pred_mean = pred_mean/counter
    targ_mean = targ_mean/counter
    print('------------------------------------------------------------')
    print(f'{identifier} Validation -> Epoch: {current_epoch}')
    print(
        f'Mean Targ LS: {targ_mean:.5f}         Mean Pred LS: {pred_mean:.5f}')
    return avg_loss


def train_HYBRID(loader, model, optimizer, criterion, scaler, identifier='', current_epoch=''):
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


def valid_HYBRID(loader, model, criterion, scaler, identifier='', current_epoch=''):
    # BRIEF: The train function completes one epoch of the training cycle.
    # PARAMETERS:
    # loader - object of PyTorch-type DataLoader to automatically feed dataset
    # model - the model to be trained
    # optimizer - the optimization algorithm applied during training
    # criterion - the loss function applied to quantify the error
    # scaler -
    # start_time = time.time()

    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    epoch_loss = 0
    timeline = []
    _preds = []
    _targs = []
    counter = 0
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/6_GRU_MSE/'

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        targets = targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions.float(), targets.float())
            # print('Current batch loss: ', loss.item())
            epoch_loss += loss.item()
            timeline.append(loss.item())
            _preds.append(predictions.cpu().detach().numpy())
            _targs.append(targets.cpu().detach().numpy())
            counter += 1

    compareFlowProfile3x3(
        preds=np.vstack(_preds),
        targs=np.vstack(_targs),
        model_id=identifier,
        dataset_id=current_epoch
        )

    print(np.vstack(_preds).shape)
    print(np.vstack(_preds).shape)
    # losses2file(
    #     losses=timeline,
    #     filename=f'{_file_prefix}Losses_Hybrid_{identifier}_{current_epoch}'
    # )
    avg_loss = epoch_loss/counter
    # duration = time.time() - start_time
    # print('------------------------------------------------------------')
    # print(f'{identifier} Validation -> Loss: {avg_loss:.3f}, Duration: {duration:.3f}')
    return avg_loss, predictions


def trial_0_UNET_AE(_alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'
    _model_identifier = f'LR{_alpha_string}'
    print('Initializing UNET_AE model with LR: ', _alpha)
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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_AE(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                alpha=_alpha_string,
                current_epoch=epoch+1
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


def trial_0_UNET_AE_mp():
    _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # _alphas = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
    _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    # _alpha_strings = ['0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _train_loaders, _valid_loaders = get_UNET_AE_loaders(file_names=0)

    processes = []
    counter = 0

    '''
    for idx, _alpha in enumerate(_alphas):
        trial_0_UNET_AE(
            _alpha=_alpha,
            _alpha_string=_alpha_strings[idx],
            _train_loaders=_train_loaders,
            _valid_loaders=_valid_loaders
        )

    '''
    for i in range(6):
        p = mp.Process(
            target=trial_0_UNET_AE,
            args=(_alphas[counter], _alpha_strings[counter],
                  _train_loaders, _valid_loaders,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {counter}')
        counter += 1

    for process in processes:
        process.join()
        print('Joining Process')

    return


def trial_0_error_timeline():
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'
    model_name = 'Model_UNET_AE_LR0_0005'
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model.load_state_dict(torch.load(f'{_directory}{model_name}'))
    _criterion = nn.L1Loss()
    _, valid_loaders = get_UNET_AE_loaders(file_names=-1)

    _datasets = [
        'C_3_0_T',
        'C_3_0_M',
        'C_3_0_B',
        'C_5_0_T',
        'C_5_0_M',
        'C_5_0_B'
    ]

    for idx, _loader in enumerate(valid_loaders):
        _losses = errorTimeline(
            loader=_loader,
            model=_model,
            criterion=_criterion
        )
        losses2file(
            losses=_losses,
            filename=f'{_directory}{model_name}_Valid_Error_Timeline_{_datasets[idx]}'
        )

    pass


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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')

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
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
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
        counter = 1

        for _rnn_depth in _rnn_depths:
            processes = []

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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')
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
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}.')
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
    # _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()
    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=0, sequence_length=5)
    # _t_loader_15, _v_loader_15 = get_RNN_loaders(file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)
    # _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    # _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]
    _R_alphas = [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]  # , 0.001]
    _R_alpha_strings = ['0_000005', '0_00001', '0_00005',
                        '0_0001', '0_0005', '0_001']  # , '0_001']
    _R_depths = [1, 1, 1, 1, 1, 1]  # , 2]
    _R_lengths = [25, 25, 25, 25, 25, 25]  # , 5]

    processes = []
    counter = 1

    for i, _a_string in enumerate(_R_alpha_strings):
        p = mp.Process(
            target=trial_2_GRU,
            args=(_R_lengths[i], _R_depths[i], _R_alphas[i],
                  _a_string, _t_loader_25, _v_loader_25,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {counter}')
        counter += 1

    p = mp.Process(
        target=trial_2_GRU,
        args=(5, 2, 0.001, '0_001', _t_loader_05, _v_loader_05,)
    )
    p.start()
    processes.append(p)
    print(f'Creating Process Number: {counter}')

    for process in processes:
        process.join()
        print('Joining Process')

    '''
    for idx, _lr in enumerate(_alphas):
        counter = 1

        for _rnn_depth in _rnn_depths:
            processes = []

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
    '''


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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')
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
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
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
        counter = 1

        for _rnn_depth in _rnn_depths:
            processes = []

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


def trial_4_Hybrid(_train_loaders, _valid_loaders, _model_rnn, _model_identifier):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/4_Hybrid/'
    # _model_identifier = 'LSTM_LR0_000005_Lay3_Seq25'
    print('Initializing model.')

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/Model_UNET_AE_LR0_0005'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=25
    ).to(device)

    _scaler = torch.cuda.amp.GradScaler()
    # _optimizer = optim.Adam(_model_hybrid.parameters(), lr=0.001)

    _train_loss = 0
    counter = 0

    for _loader in _train_loaders:
        loss, _ = valid_HYBRID(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=counter
        )
        _train_loss += loss
        resetPipeline(_model_hybrid)
        counter += 1

    print('------------------------------------------------------------')
    print(f'{_model_identifier} Training -> Averaged Loader Loss: {_train_loss/len(_train_loaders)}')

    _valid_loss = 0

    for _loader in _valid_loaders:
        loss, _ = valid_HYBRID(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=counter
        )
        _valid_loss += loss
        resetPipeline(_model_hybrid)
        counter += 1

    print('------------------------------------------------------------')
    print(f'{_model_identifier} Validation -> Averaged Loader Loss: {_valid_loss/len(_valid_loaders)}')

    # torch.save(_model_hybrid.state_dict(), f'{_file_prefix}Model_Hybrid_{_model_identifier}')

    pass


def trial_4_Hybrid_mp():
    _train_loaders, _valid_loaders = get_Hybrid_loaders(file_names=-1)
    _models = []
    _model_identifiers = [
        'RNN_LR0_00001_Lay1_Seq25',
        'GRU_LR0_00001_Lay2_Seq25',
        'LSTM_LR0_00001_Lay2_Seq25',
    ]

    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=1,
        device=device
    )
    _model_rnn_1.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_RNN/Model_RNN_LR0_00001_Lay1_Seq25'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/2_GRU/Model_GRU_LR0_00001_Lay2_Seq25'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_LSTM/Model_LSTM_LR0_00001_Lay2_Seq25'))
    _models.append(_model_rnn_3)

    counter = 1
    processes = []
    for i in range(3):
        p = mp.Process(
            target=trial_4_Hybrid,
            args=(_train_loaders, _valid_loaders,
                  _models[i], _model_identifiers[i],)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {counter}')
        counter += 1

    for process in processes:
        process.join()
        print('Joining Process')

    return


def trial_4_error_timeline():
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/'
    _unet_name = 'Model_UNET_AE_LR0_0005'
    model_name_1 = 'Model_RNN_LR0_00001_Lay1_Seq25'
    model_name_2 = 'Model_GRU_LR0_00001_Lay2_Seq25'
    model_name_3 = 'Model_LSTM_LR0_00001_Lay2_Seq25'

    _models = []
    _hybrid_models = []
    _criterion = nn.L1Loss()
    _error_timelines = [[], [], [], [], [], []]

    _model_identifiers = [
        'RNN Hybrid',
        'GRU Hybrid',
        'LSTM Hybrid',
    ]
    _dataset_identifiers = [
            'C 3 0 T',
            'C 3 0 M',
            'C 3 0 B',
            'C 5 0 T',
            'C 5 0 M',
            'C 5 0 B'
        ]
    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        f'{_directory}0_UNET_AE/{_unet_name}'))

    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=1,
        device=device
    )
    _model_rnn_1.load_state_dict(torch.load(
        f'{_directory}1_RNN/{model_name_1}'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
        f'{_directory}2_GRU/{model_name_2}'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
        f'{_directory}3_LSTM/{model_name_3}'))
    _models.append(_model_rnn_3)

    _train_loaders, _valid_loaders = get_Hybrid_loaders(file_names=-1)

    for i in range(3):
        _model_hybrid = Hybrid_MD_RNN_UNET(
            device=device,
            UNET_Model=_model_unet,
            RNN_Model=_models[i],
            seq_length=25
        ).to(device)
        _hybrid_models.append(_model_hybrid)

    for i, _loader in enumerate(_valid_loaders):
        for _model in _hybrid_models:
            error_timeline = errorTimeline(
                loader=_loader,
                model=_model,
                criterion=_criterion
            )
            _error_timelines[i].append(error_timeline)

    compareErrorTimeline_np(
        l_of_l_losses=_error_timelines,
        l_of_l_labels=_model_identifiers,
        l_of_titles=_dataset_identifiers,
        file_prefix=f'{_directory}/4_Hybrid/',
        file_name='Hybrid_Models'
    )

    pass


def trial_5_KVS_AE(_alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/'
    _model_identifier = f'LR{_alpha_string}'
    print('Initializing UNET_AE model with LR: ', _alpha)
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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_AE(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                alpha=_alpha_string,
                current_epoch=epoch+1
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


def trial_5_KVS_AE_helper():
    _t_loaders, _v_loaders = get_UNET_AE_loaders(file_names=-2)
    trial_5_KVS_AE(
        _alpha=0.0005,
        _alpha_string='0_0005',
        _train_loaders=_t_loaders,
        _valid_loaders=_v_loaders
    )


def trial_5_KVS_AE_latentspace_helper():
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    #TO DO - Check proper model to load
    _model.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/Model_UNET_AE_LR0_0005'))
    _model.eval()

    _loader_1, _loader_2_ = get_UNET_AE_loaders(
        file_names=-2,
        num_workers=1,
        batch_size=32,
        shuffle=False
    )
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/Latentspace_Dataset'
    _out_file_names = [
        '_kvs_10K_NE',
        '_kvs_10K_NW',
        '_kvs_10K_SE',
        '_kvs_10K_SW',
        '_kvs_20K_NE',
        '_kvs_20K_NW',
        '_kvs_20K_SE',
        '_kvs_20K_SW',
        '_kvs_30K_NE',
        '_kvs_30K_NW',
        '_kvs_30K_SE',
        '_kvs_30K_SW',
        '_kvs_40K_NE',
        '_kvs_40K_NW',
        '_kvs_40K_SE',
        '_kvs_40K_SW',
    ]
    for idx, _loader in enumerate(_loaders):
        get_latentspace_AE(
            loader=_loader,
            model=_model,
            out_file_name=f'{_out_directory}{_out_file_names[idx]}'
        )


def trial_5_0_KVS_error_timeline():
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/'
    model_name_1 = 'Model_UNET_AE_LR0_0005'
    model_name_2 = 'Model_LSTM_LR0_00001_Lay3_Seq25'
    model_name_3 = 'Hybrid_Model'

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    ).to(device)
    _model_unet.load_state_dict(torch.load(f'{_directory}{model_name_1}'))

    _model_rnn = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=3,
        device=device
    )
    _model_rnn.load_state_dict(torch.load(f'{_directory}{model_name_2}'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=25
    ).to(device)

    _criterion = nn.L1Loss()
    _, valid_loaders = get_Hybrid_loaders(file_names=-2)

    _datasets = [
        'kvs_40K_NE',
        'kvs_40K_NW',
        'kvs_40K_SE',
        'kvs_40K_SW'
    ]

    for idx, _loader in enumerate(valid_loaders):
        _losses = errorTimeline(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion
        )
        losses2file(
            losses=_losses,
            filename=f'{_directory}{model_name_3}_KVS_Valid_Error_Timeline_{_datasets[idx]}'
        )

    pass


def trial_5_1_KVS_RNN(_seq_length, _num_layers, _alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_KVS/'
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing RNN model.')
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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')

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


def trial_5_1_KVS_RNN_mp():
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders(
        file_names=-2, sequence_length=5)
    _t_loader_15, _v_loader_15 = get_RNN_loaders(
        file_names=-2, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=-2, sequence_length=25)

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for idx in range(1, 2):
        counter = 1

        for _rnn_depth in _rnn_depths:
            processes = []

            for i in range(3):
                p = mp.Process(
                    target=trial_5_1_KVS_RNN,
                    args=(_seq_lengths[i], _rnn_depth, _alphas[idx],
                          _alpha_strings[idx], _t_loaders[i], _v_loaders[i],)
                )
                p.start()
                processes.append(p)
                print(f'Creating Process Number: {counter}')
                counter += 1

        for process in processes:
            process.join()
            print('Joining Process')


def trial_6_GRU_MSE(_seq_length, _num_layers, _alpha, _alpha_string, _train_loaders, _valid_loaders):
    _criterion = nn.MSELoss()
    # _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/6_GRU_MSE/'
    _model_identifier = f'LR{_alpha_string}_Lay{_num_layers}_Seq{_seq_length}'
    print('Initializing GRU_MSE model.')
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
    for epoch in range(50):
        avg_loss = 0
        start_time = time.time()
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
        duration = time.time() - start_time
        print('------------------------------------------------------------')
        print(
            f'{_model_identifier} Training Epoch: {epoch+1}-> Averaged Loader Loss: {avg_loss/len(_train_loaders):.3f}. Duration: {duration:.3f}')
        _epoch_losses.append(avg_loss/len(_train_loaders))

        avg_valid = 0
        for _valid_loader in _valid_loaders:
            avg_valid += valid_RNN(
                loader=_valid_loader,
                model=_model,
                criterion=_criterion,
                scaler=_scaler,
                identifier=_model_identifier,
                current_epoch=epoch+1
            )
        print('------------------------------------------------------------')
        print(f'{_model_identifier} Validation -> Averaged Loader Loss: {avg_valid/len(_valid_loaders):.3f}.')
        _epoch_valids.append(avg_valid/len(_valid_loaders))

    losses2file(
        losses=_epoch_losses,
        filename=f'{_file_prefix}Losses_GRU_MSE_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        filename=f'{_file_prefix}Valids_GRU_MSE_{_model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_GRU_MSE_{_model_identifier}.csv',
            f'{_file_prefix}Valids_GRU_MSE_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_GRU_MSE_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_GRU_MSE_{_model_identifier}'
    )


def trial_6_GRU_MSE_mp():
    # _alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # _alpha_strings = ['0_01', '0_005', '0_001', '0_0005', '0_0001', '0_00005']
    # _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    # _alpha_strings = ['0_001', '0_0005', '0_0001', '0_00005', '0_00001', '0_000005']
    _alphas = [0.0001, 0.00005, 0.00001, 0.000005]
    _alpha_strings = ['0_0001', '0_00005', '0_00001', '0_000005']
    _rnn_depths = [1, 3]
    _seq_lengths = [25]

    _alphas.reverse()
    _alpha_strings.reverse()
    # _t_loader_05, _v_loader_05 = get_RNN_loaders(file_names=0, sequence_length=5)
    # _t_loader_15, _v_loader_15 = get_RNN_loaders(file_names=0, sequence_length=15)
    _t_loader_25, _v_loader_25 = get_RNN_loaders(
        file_names=0, sequence_length=25)
    # _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    # _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]
    _t_loaders = [_t_loader_25]
    _v_loaders = [_v_loader_25]

    counter = 1

    for idx, _lr in enumerate(_alphas):
        processes = []
        for _rnn_depth in _rnn_depths:
            p = mp.Process(
                target=trial_6_GRU_MSE,
                args=(_seq_lengths[0], _rnn_depth, _lr,
                      _alpha_strings[idx], _t_loaders[0], _v_loaders[0],)
            )
            p.start()
            processes.append(p)
            print(f'Creating Process Number: {counter}')
            counter += 1

        for process in processes:
            process.join()
            print('Joining Process')


def trial_6_flow_profile():
    _criterion = nn.MSELoss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/6_GRU_MSE/'
    _pwd = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/'
    _model_identifier = 'Hybrid_UNET_AE_GRU_L3_S25'
    print('Initializing model.')

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        f'{_pwd}0_UNET_AE/Model_UNET_AE_LR0_0005'))

    _model_rnn = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=3,
        device=device
    )
    _model_rnn.load_state_dict(torch.load(
            f'{_pwd}6_GRU_MSE/Model_GRU_MSE_LR0_00001_Lay3_Seq25'))

    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=_model_rnn,
        seq_length=25
    ).to(device)

    _scaler = torch.cuda.amp.GradScaler()
    # _optimizer = optim.Adam(_model_hybrid.parameters(), lr=0.001)

    _train_loss = 0
    counter = 0
    _train_loaders, _valid_loaders = get_Hybrid_loaders(file_names=-1)

    for _loader in _train_loaders:
        loss, _ = valid_HYBRID(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=counter
        )
        _train_loss += loss
        resetPipeline(_model_hybrid)
        counter += 1

    print('------------------------------------------------------------')
    print(f'{_model_identifier} Training -> Averaged Loader Loss: {_train_loss/len(_train_loaders):.6f}')

    _valid_loss = 0

    for _loader in _valid_loaders:
        loss, _ = valid_HYBRID(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            scaler=_scaler,
            identifier=_model_identifier,
            current_epoch=counter
        )
        _valid_loss += loss
        resetPipeline(_model_hybrid)
        counter += 1

    print('------------------------------------------------------------')
    print(f'{_model_identifier} Validation -> Averaged Loader Loss: {_valid_loss/len(_valid_loaders):.6f}')

    torch.save(
        _model_hybrid.state_dict(),
        f'{_file_prefix}Model_{_model_identifier}'
    )


if __name__ == "__main__":

    trial_5_KVS_AE_latentspace_helper()
