import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_UNET, resetPipeline
from utils_new import get_testing_loaders
from trial_5 import valid_HYBRID_Couette
from trial_6 import valid_HYBRID_KVS

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


def analysis_1_Couette(model_rnn, model_identifier, test_loaders):
    """The analysis_1_Couette function creates a Hybrid_MD_RNN_UNET model on the
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

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_UNET_AE/Model_UNET_AE_LR0_0005'))

    print('Initializing Hybrid_MD_RNN_UNET model.')
    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=model_rnn,
        seq_length=25
    ).to(device)

    _counter = 0

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


def analysis_1_Couette_mp():
    """The analysis_1_Couette_mp function is essentially a helper function to
    facilitate the testing of multiple concurrent models via multiprocessing
    of the analysis_1_Couette function. Here, 3 unique models are tested using
    the best performing UNET_AE (pretrained) from trial_1 and the best
    performing RNN/GRU/LSTM (pretrained) from trials_2 - trial_4.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 1: Hybrid MD RNN UNET (Couette)')
    _test_loaders = get_testing_loaders(
        data_distribution='get_couette',
        batch_size=1,
        shuffle=False
    )
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
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/2_RNN/Model_RNN_LR0_00001_Lay1_Seq25'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/3_GRU/Model_GRU_LR0_00001_Lay2_Seq25'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/4_LSTM/Model_LSTM_LR0_00001_Lay2_Seq25'))
    _models.append(_model_rnn_3)

    _processes = []
    for i in range(3):
        _p = mp.Process(
            target=analysis_1_Couette,
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


def analysis_1_KVS(model_rnn, model_identifier, test_loaders):
    """The analysis_1_KVS function creates a Hybrid_MD_RNN_UNET model on the
    basis of a trained UNET_AE and a trained RNN/GRU/LSTM. It then documents
    its performance w.r.t. time series prediction, i.e. performance in
    accurately predicting the cell velocities for the next MD timestep. This is
    done as a proof of concept merley via terminal output. In addition, this
    function calls the valid_HYBRID_KVS function which automatically
    compares prediction and target u_z values for the entire dataset. Refer to
    valid_HYBRID_KVS for more details.

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

    _model_unet = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )
    _model_unet.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
        '/Results/6_Hybrid_KVS/Model_UNET_AE_KVS_LR0_0005'))

    print('Initializing Hybrid_MD_RNN_UNET model.')
    _model_hybrid = Hybrid_MD_RNN_UNET(
        device=device,
        UNET_Model=_model_unet,
        RNN_Model=model_rnn,
        seq_length=25
    ).to(device)

    _counter = 0

    _train_loss = 0
    for _loader in test_loaders:
        _loss, _ = valid_HYBRID_KVS(
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


def analysis_1_KVS_mp():
    """The analysis_1_KVS_mp function is essentially a helper function to
    facilitate the testing of multiple concurrent models via multiprocessing
    of the analysis_1_KVS function. Here, 3 unique models are tested using
    the best performing UNET_AE (pretrained) from trial_1 and the best
    performing RNN/GRU/LSTM (pretrained) from trial_6.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Analysis 1: Hybrid MD RNN UNET (Couette)')
    _test_loaders = get_testing_loaders(
        data_distribution='get_KVS',
        batch_size=1,
        shuffle=False
    )
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
            '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach'
            '/Results/6_Hybrid_KVS/Model_KVS_RNN_LR0_00001_Lay1_Seq25'))
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
            '/Results/6_Hybrid_KVS/Model_KVS_GRU_LR0_00001_Lay2_Seq25'))
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
            '/Results/6_Hybrid_KVS/Model_KVS_LSTM_LR0_00001_Lay2_Seq25'))
    _models.append(_model_rnn_3)

    _processes = []
    for i in range(3):
        _p = mp.Process(
            target=analysis_1_KVS,
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


if __name__ == "__main__":
    analysis_1_Couette_mp()
    analysis_1_KVS_mp()
    pass
