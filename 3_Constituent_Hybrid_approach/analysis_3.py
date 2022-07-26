import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import AE, UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_AE, resetPipeline
from utils_new import get_UNET_AE_loaders, get_RNN_loaders_analysis_3, losses2file, get_Hybrid_loaders, get_testing_loaders
from plotting import compareAvgLoss, compareLossVsValid
from trial_1 import train_AE, valid_AE, get_latentspace_AE
from trial_2 import train_RNN, valid_RNN
from trial_5 import valid_HYBRID_Couette

torch.manual_seed(10)
random.seed(10)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

plt.style.use(['science'])
np.set_printoptions(precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0


def analysis_3_Couette_non_UNET(alpha, alpha_string, train_loaders, valid_loaders):
    """The analysis_3_Couette_non_UNET function trains the given model on the
    Couette data distribution. It documents model progress via saving average
    training and validation losses to file and comparing them in a plot.

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
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/AE/'
    _model_identifier = f'LR{alpha_string}'
    print('Initializing AE model.')
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
        file_name=f'{_file_prefix}Losses_AE_{_model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_AE_{_model_identifier}'
    )

    compareLossVsValid(
        loss_files=[
            f'{_file_prefix}Losses_AE_{_model_identifier}.csv',
            f'{_file_prefix}Valids_AE_{_model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'_AE_{_model_identifier}'
    )
    torch.save(
        _model.state_dict(),
        f'{_file_prefix}Model_AE_{_model_identifier}'
    )
    return


def analysis_3_Couette_non_UNET_mp():
    """The analysis_3_Couette_non_UNET_mp function is essentially a helper
    function to facilitate the training of various AE model configurations
    on the basis of the Couette data distribution.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 3: AE (Couette)')
    _t_loaders, _v_loaders = get_UNET_AE_loaders(
        data_distribution='get_couette',
        batch_size=32,
        shuffle=True
    )

    _alphas = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
    _alpha_strings = ['0_0005', '0_0001', '0_00005',
                      '0_00001', '0_000005', '0_000001']

    _processes = []

    for i in range(6):
        _p = mp.Process(
            target=analysis_3_Couette_non_UNET,
            args=(_alphas[i], _alpha_strings[i],
                  _t_loaders, _v_loaders,)
        )
        _p.start()
        _processes.append(_p)
        print(f'Creating Process Number: {i+1}')

    for _process in _processes:
        _process.join()
        print('Joining Process')
    return


def analysis_3_Couette_non_UNET_latentspace_helper():
    """The analysis_3_Couette_non_UNET_latentspace_helper function contains the
    additional steps to create the model-specific latentspace. It loads an
    already trained model in model.eval() mode, loads the dataset loaders and
    calls the get_latentspace_AE function for each individual subdataset in the
    training and validation datasets.

    Args:
        NONE

    Returns:
        NONE:
    """
    print('Starting Analysis 3: Get Latentspace (Couette)')
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                   '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/AE/'
    _model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    _model.load_state_dict(torch.load(
        f'{_file_prefix}Model_AE_LR0_00001'))
    _model.eval()

    _loader_1, _loader_2_ = get_UNET_AE_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )
    _loaders = _loader_1 + _loader_2_
    _out_directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/CleanCouette_AE_LS/Latentspace_Dataset'
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


def analysis_3_Couette_RNN(model, model_identifier, alpha, train_loaders, valid_loaders):
    """The analysis_3_Couette_RNN function trains the given model and documents its
    progress via saving average training and validation losses to file and
    comparing them in a plot.

    Args:
        model:
          Object of PyTorch Module class, i.e. the model to be trained.
        model_identifier:
          A unique string to identify the model. Here, a combination of the
          learning rate (_alpha), num of RNN layers (_num_layers) and sequence
          length (_seq_length) is used.
        alpha:
          A double value indicating the chosen learning rate.
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
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
        '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/RNNs/'

    print('Initializing training parameters.')
    _scaler = torch.cuda.amp.GradScaler()
    _optimizer = optim.Adam(model.parameters(), lr=alpha)
    _epoch_losses = []
    _epoch_valids = []

    print('Beginning training.')
    for epoch in range(250):
        _avg_loss = 0
        for _train_loader in train_loaders:
            _avg_loss += train_RNN(
                loader=_train_loader,
                model=model,
                optimizer=_optimizer,
                criterion=_criterion,
                scaler=_scaler,
                model_identifier=model_identifier,
                current_epoch=epoch+1
            )
        _avg_loss = _avg_loss/len(train_loaders)
        print('------------------------------------------------------------')
        print(
            f'{model_identifier} Training Epoch: {epoch+1}-> Averaged '
            f'Loader Loss: {_avg_loss:.3f}')

        _epoch_losses.append(_avg_loss)

        _avg_valid = 0
        for _valid_loader in valid_loaders:
            _avg_valid += valid_RNN(
                loader=_valid_loader,
                model=model,
                criterion=_criterion,
                model_identifier=model_identifier
            )
        _avg_valid = _avg_valid/len(valid_loaders)
        print('------------------------------------------------------------')
        print(f'{model_identifier} Validation -> Averaged '
              f'Loader Loss: {_avg_valid:.3f}')
        _epoch_valids.append(_avg_valid)

    losses2file(
        losses=_epoch_losses,
        file_name=f'{_file_prefix}Losses_{model_identifier}'
    )
    losses2file(
        losses=_epoch_valids,
        file_name=f'{_file_prefix}Valids_{model_identifier}'
    )

    compareAvgLoss(
        loss_files=[
            f'{_file_prefix}Losses_{model_identifier}.csv',
            f'{_file_prefix}Valids_{model_identifier}.csv'
        ],
        loss_labels=['Training', 'Validation'],
        file_prefix=_file_prefix,
        file_name=f'And_Valids_{model_identifier}'
    )
    torch.save(
        model.state_dict(),
        f'{_file_prefix}Model_{model_identifier}'
    )


def analysis_3_Couette_RNN_mp():
    """The analysis_3_Couette_RNN_mp function is essentially a helper function
    to facilitate the training of multiple concurrent models via multiprocessing
    of the analysis_3_Couette_RNN function. Here, 3*54 unique models are trained
    using various RNN/GRU/LSTM configurations. Refer to the analysis_3_Couette_RNN
    function for more details.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 3: RNN/GRU/LSTM (Couette)')
    _alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]  # , 0.000005]
    _alpha_strings = ['0_001', '0_0005', '0_0001',
                      '0_00005', '0_00001']  # , '0_000005']
    _rnn_depths = [1, 2, 3]
    _seq_lengths = [5, 15, 25]

    _alphas.reverse()
    _alpha_strings.reverse()

    _t_loader_05, _v_loader_05 = get_RNN_loaders_analysis_3(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=5,
        shuffle=True
    )
    _t_loader_15, _v_loader_15 = get_RNN_loaders_analysis_3(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=15,
        shuffle=True
    )
    _t_loader_25, _v_loader_25 = get_RNN_loaders_analysis_3(
        data_distribution='get_couette',
        batch_size=32,
        seq_length=25,
        shuffle=True
    )

    _t_loaders = [_t_loader_05, _t_loader_15, _t_loader_25]
    _v_loaders = [_v_loader_05, _v_loader_15, _v_loader_25]

    for i, _lr in enumerate(_alpha_strings):

        for j in _rnn_depths:
            _processes = []
            _counter = 1
            for k, seq in enumerate(_seq_lengths):
                _model_rnn_1 = RNN(
                    input_size=256,
                    hidden_size=256,
                    seq_size=seq,
                    num_layers=j,
                    device=device
                ).to(device)
                _model_id_1 = f'RNN_LR{_lr}_Lay{j}_Seq{seq}'
                _p1 = mp.Process(
                    target=analysis_3_Couette_RNN,
                    args=(_model_rnn_1, _model_id_1,
                          _alphas[i], _t_loaders[k], _v_loaders[k],)
                )
                _p1.start()
                _processes.append(_p1)
                print(f'Creating Process Number: {_counter} for {_model_id_1}')
                _counter += 1
                ###################
                _model_rnn_2 = GRU(
                    input_size=256,
                    hidden_size=256,
                    seq_size=seq,
                    num_layers=j,
                    device=device
                ).to(device)
                _model_id_2 = f'GRU_LR{_lr}_Lay{j}_Seq{seq}'
                _p2 = mp.Process(
                    target=analysis_3_Couette_RNN,
                    args=(_model_rnn_2, _model_id_2,
                          _alphas[i], _t_loaders[k], _v_loaders[k],)
                )
                _p2.start()
                _processes.append(_p2)
                print(f'Creating Process Number: {_counter} for {_model_id_2}')
                _counter += 1
                ###################
                _model_rnn_3 = LSTM(
                    input_size=256,
                    hidden_size=256,
                    seq_size=seq,
                    num_layers=j,
                    device=device
                ).to(device)
                _model_id_3 = f'LSTM_LR{_lr}_Lay{j}_Seq{seq}'
                _p3 = mp.Process(
                    target=analysis_3_Couette_RNN,
                    args=(_model_rnn_3, _model_id_3,
                          _alphas[i], _t_loaders[k], _v_loaders[k],)
                )
                _p3.start()
                _processes.append(_p3)
                print(f'Creating Process Number: {_counter} for {_model_id_3}')
                _counter += 1

            for _process in _processes:
                _process.join()
                print('Joining Process')


def analysis_3_Couette_Hybrid(model_rnn, model_identifier, seq_length, train_loaders, valid_loaders):
    """The analysis_3_Couette_Hybrid function creates a Hybrid_MD_RNN_UNET model
    on the basis of a trained UNET_AE and a trained RNN/GRU/LSTM. It then documents
    its performance w.r.t. time series prediction, i.e. performance in accurately
    predicting the cell velocities for the next MD timestep. This is done as a
    proof of concept merley via terminal output. In addition, this function calls
    the valid_HYBRID_Couette function which automatically compares flow profiles
    of model prediction and corresponding target via the plotPredVsTargCouette
    function. Refer to valid_HYBRID_Couette for more details.

    Args:
        model_rnn:
          Object of PyTorch Module class, i.e. the RNN/GRU/LSTM model to be
          incorporated into the hybrid model.
        model_identifier:
          A unique string to identify the model. Here the RNN configuration is
          used to identify the RNN model (RNN-Type/LR0_XXXLayX_SeqXX)
        train_loaders:
          Object of PyTorch-type DataLoader to automatically pass training
          dataset to model.
        valid_loaders:
          Object of PyTorch-type DataLoader to automatically pass validation
          dataset to model.

    Returns:
        NONE:
          This function documents model progress by printing the average loss
          for each training and validation set to the terminal.
    """
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                   '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/AE/'

    _model_ae = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_ae.load_state_dict(torch.load(
        f'{_file_prefix}Model_AE_LR0_0001'))

    print('Initializing Hybrid_MD_RNN_AE model.')
    _model_hybrid = Hybrid_MD_RNN_AE(
        device=device,
        AE_Model=_model_ae,
        RNN_Model=model_rnn,
        seq_length=seq_length
    ).to(device)

    _counter = 0

    _train_loss = 0
    for _loader in train_loaders:
        _loss, _ = valid_HYBRID_Couette(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            model_identifier=model_identifier,
            dataset_identifier=str(_counter)
        )
        _train_loss += _loss
        resetPipeline(_model_hybrid)
        _counter += 1

    print('------------------------------------------------------------')
    print(f'{model_identifier} Training -> Averaged Loader Loss: '
          f'{_train_loss/len(train_loaders)}')

    _valid_loss = 0
    for _loader in valid_loaders:
        _loss, _ = valid_HYBRID_Couette(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            model_identifier=model_identifier,
            dataset_identifier=str(_counter)
        )
        _valid_loss += _loss
        resetPipeline(_model_hybrid)
        _counter += 1

    print('------------------------------------------------------------')
    print(f'{model_identifier} Validation -> Averaged Loader Loss: '
          f'{_valid_loss/len(valid_loaders)}')
    return


def analysis_3_Couette_Hybrid_mp():
    """The analysis_3_Couette_Hybrid_mp function is essentially a helper function
    to facilitate the validation of multiple concurrent models via multiprocessing
    of the analysis_3_Couette_Hybrid function. Here, 3 unique models are validated
    using the best performing UNET_AE (pretrained) from trial_1 and the best
    performing RNN/GRU/LSTM (pretrained) from analysis_3_Couette_RNN_mp.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 3: Hybrid MD RNN AE Train./Valid. (Couette)')

    _models = []
    _model_identifiers = [
        'RNN_LR0_0001_Lay1_Seq25',
        'GRU_LR0_0001_Lay2_Seq25',
        'LSTM_LR0_0001_Lay2_Seq25',
    ]
    _seq_lengths = [25, 25, 25]
    _num_layers = [1, 2, 2]
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                   '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/RNNs/'
    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[0],
        num_layers=_num_layers[0],
        device=device
    )
    _model_rnn_1.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[0]}'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[1],
        num_layers=_num_layers[1],
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[1]}'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[2],
        num_layers=_num_layers[2],
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[2]}'))
    _models.append(_model_rnn_3)

    _train_loaders, _valid_loaders = get_Hybrid_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )

    _processes = []
    for i in range(3):
        _p = mp.Process(
            target=analysis_3_Couette_Hybrid,
            args=(_models[i], _model_identifiers[i], _seq_lengths[i],
                  _train_loaders, _valid_loaders,)
        )
        _p.start()
        _processes.append(_p)
        print(f'Creating Process Number: {i+1}')

    for _process in _processes:
        _process.join()
        print('Joining Process')
    return


def analysis_3_Couette_Test(model_rnn, model_identifier, test_loaders):
    """The analysis_3_Couette_test function creates a Hybrid_MD_RNN_UNET modelon the
    basis of a trained UNET_AE and a trained RNN/GRU/LSTM. It then documents
    its performance w.r.t. time series prediction, i.e. performance in
    accurately predicting the cell velocities for the next MD timestep. This is
    done as a proof of concept merley via terminal output. In addition, this
    function calls the valid_HYBRID_Couette function which automatically
    compares flow profiles of model prediction and corresponding target via the
    compareFlowProfile3x3 function. Refer to valid_HYBRID_Couette for more
    details.

    Args:
        model_rnn:
          Object of PyTorch Module class, i.e. the RNN/GRU/LSTM model to be
          incorporated into the hybrid model.
        model_identifier:
          A unique string to identify the model. Here the RNN configuration is
          used to identify the RNN model (RNN-Type/LR0_XXXLayX_SeqXX)
        test_loaders:
          Object of PyTorch-type DataLoader to automatically pass testing
          dataset to model.

    Returns:
        NONE:
          This function documents model progress by printing the average loss
          for each training and validation set to the terminal.
    """
    _criterion = nn.L1Loss()
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                   '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/AE/'

    _model_ae = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_ae.load_state_dict(torch.load(
        f'{_file_prefix}Model_AE_LR0_0001'))

    print('Initializing Hybrid_MD_RNN_UNET model.')
    _model_hybrid = Hybrid_MD_RNN_AE(
        device=device,
        AE_Model=_model_ae,
        RNN_Model=model_rnn,
        seq_length=25
    ).to(device)

    _counter = 18

    _train_loss = 0
    for _loader in test_loaders:
        _loss, _ = valid_HYBRID_Couette(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            model_identifier=model_identifier,
            dataset_identifier=_counter
        )
        _train_loss += _loss
        resetPipeline(_model_hybrid)
        _counter += 1

    print('------------------------------------------------------------')
    print(f'{model_identifier} Training -> Averaged Loader Loss: '
          f'{_train_loss/len(test_loaders)}')

    return


def analysis_3_Couette_Test_mp():
    """The analysis_3_Couette_mp function is essentially a helper function to
    facilitate the testing of multiple concurrent models via multiprocessing
    of the analysis_3_Couette function. Here, 3 unique models are tested using
    the best performing UNET_AE (pretrained) from trial_1 and the best
    performing RNN/GRU/LSTM (pretrained) from trials_2 - trial_4.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 3: Hybrid MD RNN AE Test. (Couette)')

    _models = []
    _model_identifiers = [
        'RNN_LR0_0001_Lay1_Seq25',
        'GRU_LR0_0001_Lay2_Seq25',
        'LSTM_LR0_0001_Lay2_Seq25',
    ]
    _seq_lengths = [25, 25, 25]
    _num_layers = [1, 2, 2]
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
                   '3_Constituent_Hybrid_approach/Results/9_Analysis_3_non_UNET/RNNs/'
    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[0],
        num_layers=_num_layers[0],
        device=device
    )
    _model_rnn_1.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[0]}'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[1],
        num_layers=_num_layers[1],
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[1]}'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=_seq_lengths[2],
        num_layers=_num_layers[2],
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
            f'{_file_prefix}Model_{_model_identifiers[2]}'))
    _models.append(_model_rnn_3)

    _test_loaders = get_testing_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )

    _processes = []
    for i in range(3):
        _p = mp.Process(
            target=analysis_3_Couette_Test,
            args=(_models[i], _model_identifiers[i],
                  _test_loaders)
        )
        _p.start()
        _processes.append(_p)
        print(f'Creating Process Number: {i+1}')

    for _process in _processes:
        _process.join()
        print('Joining Process')
    return


def rubbish():
    '''
    def trial_7_Hybrid_KVS_RNN(model, model_identifier, alpha, train_loaders, valid_loaders):
        """The trial_7_Hybrid_KVS_RNN function trains the given model and documents
        its progress via saving average training and validation losses to file and
        comparing them in a plot.

        Args:
            model:
              Object of PyTorch Module class, i.e. the model to be trained.
            model_identifier:
              A unique string to identify the model. Here, a combination of the
              learning rate (_alpha), num of RNN layers (_num_layers) and sequence
              length (_seq_length) is used.
            alpha:
              A double value indicating the chosen learning rate.
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
        _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/' + \
            '3_Constituent_Hybrid_approach/Results/7_Hybrid_KVS_non_UNET/'

        print('Initializing training parameters.')
        _scaler = torch.cuda.amp.GradScaler()
        _optimizer = optim.Adam(model.parameters(), lr=alpha)
        _epoch_losses = []
        _epoch_valids = []

        print('Beginning training.')
        for epoch in range(150):
            _avg_loss = 0
            for _train_loader in train_loaders:
                _avg_loss += train_RNN(
                    loader=_train_loader,
                    model=model,
                    optimizer=_optimizer,
                    criterion=_criterion,
                    scaler=_scaler,
                    model_identifier=model_identifier,
                    current_epoch=epoch+1
                )
            _avg_loss = _avg_loss/len(train_loaders)
            print('------------------------------------------------------------')
            print(
                f'{model_identifier} Training Epoch: {epoch+1}-> Averaged '
                f'Loader Loss: {_avg_loss:.3f}')

            _epoch_losses.append(_avg_loss)

            _avg_valid = 0
            for _valid_loader in valid_loaders:
                _avg_valid += valid_RNN(
                    loader=_valid_loader,
                    model=model,
                    criterion=_criterion,
                    model_identifier=model_identifier
                )
            _avg_valid = _avg_valid/len(valid_loaders)
            print('------------------------------------------------------------')
            print(f'{model_identifier} Validation -> Averaged '
                  f'Loader Loss: {_avg_valid:.3f}')
            _epoch_valids.append(_avg_valid)

        losses2file(
            losses=_epoch_losses,
            file_name=f'{_file_prefix}Losses_{model_identifier}'
        )
        losses2file(
            losses=_epoch_valids,
            file_name=f'{_file_prefix}Valids_{model_identifier}'
        )

        compareAvgLoss(
            loss_files=[
                f'{_file_prefix}Losses_{model_identifier}.csv',
                f'{_file_prefix}Valids_{model_identifier}.csv'
            ],
            loss_labels=['Training', 'Validation'],
            file_prefix=_file_prefix,
            file_name=f'And_Valids_{model_identifier}'
        )
        torch.save(
            model.state_dict(),
            f'{_file_prefix}Model_{model_identifier}'
        )


    def trial_7_Hybrid_KVS_RNN_mp():
        """The trial_7_Hybrid_KVS_RNN_mp function is essentially a helper function
        to facilitate the training of multiple concurrent models via multiprocessing
        of the trial_7_Hybrid_KVS_RNN function. Here, 3 unique models are trained using
        the most promising RNN/GRU/LSTM configurations from trials 2/3/4. Refer to
        the trial_7_Hybrid_KVS_RNN function for more details.

        Args:
            NONE

        Returns:
            NONE
        """
        print('Starting Trial 7: RNN_mp (KVS, AE)')
        _models = []
        _model_identifiers = [
            'KVS_AE_RNN_LR0_00001_Lay1_Seq25',
            'KVS_AE_GRU_LR0_00001_Lay2_Seq25',
            'KVS_AE_LSTM_LR0_00001_Lay2_Seq25',
        ]
        _alphas = [0.00001, 0.00001, 0.00001]

        _model_rnn_1 = RNN(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=1,
            device=device
        ).to(device)
        _models.append(_model_rnn_1)
        _model_rnn_2 = GRU(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=2,
            device=device
        ).to(device)
        _models.append(_model_rnn_2)
        _model_rnn_3 = LSTM(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=2,
            device=device
        ).to(device)
        _models.append(_model_rnn_3)

        _t_loader_25, _v_loader_25 = get_RNN_loaders(
            data_distribution="get_AE_KVS",
            batch_size=32,
            seq_length=25,
            shuffle=True
        )
        processes = []

        for i in range(3):
            p = mp.Process(
                target=trial_7_Hybrid_KVS_RNN,
                args=(_models[i], _model_identifiers[i], _alphas[i],
                      _t_loader_25, _v_loader_25,)
            )
            p.start()
            processes.append(p)
            print(f'Creating Process Number: {i+1}')

        for process in processes:
            process.join()
            print('Joining Process')


    def trial_7_KVS_Hybrid(model_rnn, model_identifier, train_loaders, valid_loaders):
        """The trial_7_KVS_Hybrid function creates a Hybrid_MD_RNN_UNET model on the
        basis of a trained AE and a trained RNN/GRU/LSTM. It then documents
        its performance w.r.t. time series prediction, i.e. performance in
        accurately predicting the cell velocities for the next MD timestep. This is
        done as a proof of concept merley via terminal output. In addition, this
        function calls the valid_HYBRID_KVS function which automatically compares
        flow profiles of model prediction and corresponding target via the
        plotVelocityField function. Refer to valid_HYBRID_KVS for more details.

        Args:
            model_rnn:
              Object of PyTorch Module class, i.e. the RNN/GRU/LSTM model to be
              incorporated into the hybrid model.
            model_identifier:
              A unique string to identify the model. Here the RNN configuration is
              used to identify the RNN model (RNN-Type/LR0_XXXLayX_SeqXX)
            train_loaders:
              Object of PyTorch-type DataLoader to automatically pass training
              dataset to model.
            valid_loaders:
              Object of PyTorch-type DataLoader to automatically pass validation
              dataset to model.

        Returns:
            NONE:
              This function documents model progress by printing the average loss
              for each training and validation set to the terminal.
        """
        _criterion = nn.L1Loss()

        _model_AE = AE(
            device=device,
            in_channels=3,
            out_channels=3,
            features=[4, 8, 16],
            activation=nn.ReLU(inplace=True)
        )
        _model_AE.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
            '/Results/7_Hybrid_KVS_non_UNET/Model_AE_KVS_LR0_0001'))

        print('Initializing Hybrid_MD_RNN_UNET model.')
        _model_hybrid = Hybrid_MD_RNN_AE(
            device=device,
            AE_Model=_model_AE,
            RNN_Model=model_rnn,
            seq_length=25
        ).to(device)

        _counter = 0

        _train_loss = 0
        for _loader in train_loaders:
            _loss, _ = valid_HYBRID_KVS(
                loader=_loader,
                model=_model_hybrid,
                criterion=_criterion,
                model_identifier=model_identifier,
                dataset_identifier=str(_counter)
            )
            _train_loss += _loss
            resetPipeline(_model_hybrid)
            _counter += 1

        print('------------------------------------------------------------')
        print(f'{model_identifier} Training -> Averaged Loader Loss: '
              f'{_train_loss/len(train_loaders)}')

        _valid_loss = 0
        for _loader in valid_loaders:
            _loss, _ = valid_HYBRID_KVS(
                loader=_loader,
                model=_model_hybrid,
                criterion=_criterion,
                model_identifier=model_identifier,
                dataset_identifier=str(_counter)
            )
            _valid_loss += _loss
            resetPipeline(_model_hybrid)
            _counter += 1

        print('------------------------------------------------------------')
        print(f'{model_identifier} Validation -> Averaged Loader Loss: '
              f'{_valid_loss/len(valid_loaders)}')
        return


    def trial_7_KVS_Hybrid_mp():
        """The trial_7_KVS_Hybrid_mp function is essentially a helper function to
        facilitate the validation of multiple concurrent models via multiprocessing
        of the trial_7_KVS_Hybrid function. Here, 3 unique models are validated using
        the best performing AE (pretrained) from trial_1 and the best performing
        RNN/GRU/LSTM (pretrained) models.

        Args:
            NONE

        Returns:
            NONE
        """
        print('Starting Trial 6: Hybrid MD RNN AE (KVS, AE)')
        _train_loaders, _valid_loaders = get_Hybrid_loaders(
            data_distribution='get_KVS',
            batch_size=1,
            shuffle=False
        )
        _models = []
        _model_identifiers = [
            'AE_RNN_LR0_00001_Lay1_Seq25',
            'AE_GRU_LR0_00001_Lay2_Seq25',
            'AE_LSTM_LR0_00001_Lay2_Seq25',
        ]

        _model_rnn_1 = RNN(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=1,
            device=device
        )
        _model_rnn_1.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
            '/Results/7_Hybrid_KVS_non_UNET/Model_KVS_AE_RNN_LR0_00001_Lay1_Seq25'))
        _models.append(_model_rnn_1)

        _model_rnn_2 = GRU(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=2,
            device=device
        )
        _model_rnn_2.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
            '/Results/7_Hybrid_KVS_non_UNET/Model_KVS_AE_GRU_LR0_00001_Lay2_Seq25'))
        _models.append(_model_rnn_2)

        _model_rnn_3 = LSTM(
            input_size=256,
            hidden_size=256,
            seq_size=25,
            num_layers=2,
            device=device
        )
        _model_rnn_3.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
            '/Results/7_Hybrid_KVS_non_UNET/Model_KVS_AE_LSTM_LR0_00001_Lay2_Seq25'))
        _models.append(_model_rnn_3)

        _processes = []
        for i in range(3):
            _p = mp.Process(
                target=trial_7_KVS_Hybrid,
                args=(_models[i], _model_identifiers[i],
                      _train_loaders, _valid_loaders,)
            )
            _p.start()
            _processes.append(_p)
            print(f'Creating Process Number: {i+1}')

        for _process in _processes:
            _process.join()
            print('Joining Process')
        return
    '''
    pass


if __name__ == "__main__":
    analysis_3_Couette_RNN_mp()
