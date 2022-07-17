import torch
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from model import UNET_AE, RNN, GRU, LSTM, Hybrid_MD_RNN_UNET, resetPipeline
from utils_new import get_Hybrid_loaders
from trial_1 import error_timeline
from plotting import compareFlowProfile3x3, compareErrorTimeline_np, plotPredVsTargCouette

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


def valid_HYBRID_Couette(loader, model, criterion, model_identifier, dataset_identifier):
    """The valid_Hybrid function computes the average loss on a given dataset
    without updating/optimizing the learnable model parameters. Additionally,
    it passes two stacked numpy arrays containing predictions and targets,
    respectively, to the compareFlowProfile3x3 function to create and save
    graphical comparisons.

    Args:
        loader:
          Object of PyTorch-type DataLoader to automatically feed dataset
        model:
          Object of PyTorch MOdule class, i.e. the model to be trained.
        criterion:
          The loss function applied to quantify the error.
        model_identifier:
          A unique string to identify the model. Here, the learning rate is
          used to identify which model is being trained.

    Returns:
        avg_loss:
          A double value indicating average validation loss for the current epoch.
        predictions:
          A list of numpy arrays containing model predictions.
    """
    _file_prefix = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_Couette'
    _epoch_loss = 0
    _timeline = []
    _preds = []
    _targs = []
    _counter = 0

    for batch_idx, (_data, _targets) in enumerate(loader):
        _data = _data.float().to(device=device)
        _targets = _targets.float().to(device=device)

        with torch.cuda.amp.autocast():
            _predictions = model(_data)
            _loss = criterion(_predictions.float(), _targets.float())
            _epoch_loss += _loss.item()
            _timeline.append(_loss.item())
            _preds.append(_predictions.cpu().detach().numpy())
            _targs.append(_targets.cpu().detach().numpy())
            _counter += 1
    '''
    compareFlowProfile3x3(
        preds=np.vstack(_preds),
        targs=np.vstack(_targs),
        model_id=model_identifier,
        dataset_id=dataset_identifier
    )
    '''
    plotPredVsTargCouette(
        input_1=np.vstack(_preds),
        input_2=np.vstack(_targs),
        file_prefix=_file_prefix,
        file_name=model_identifier+str(dataset_identifier)
    )

    _avg_loss = _epoch_loss/_counter
    return _avg_loss, _predictions


def trial_5_Hybrid(model_rnn, model_identifier, train_loaders, valid_loaders):
    """The trial_5_Hybrid function creates a Hybrid_MD_RNN_UNET model on the
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
    for _loader in train_loaders:
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
          f'{_train_loss/len(train_loaders)}')

    _valid_loss = 0
    for _loader in valid_loaders:
        _loss, _ = valid_HYBRID_Couette(
            loader=_loader,
            model=_model_hybrid,
            criterion=_criterion,
            model_identifier=model_identifier,
            dataset_identifier=_counter
        )
        _valid_loss += _loss
        resetPipeline(_model_hybrid)
        _counter += 1

    print('------------------------------------------------------------')
    print(f'{model_identifier} Validation -> Averaged Loader Loss: '
          f'{_valid_loss/len(valid_loaders)}')
    return


def trial_5_Hybrid_mp():
    """The trial_5_Hybrid_mp function is essentially a helper function to
    facilitate the validation of multiple concurrent models via multiprocessing
    of the trial_5_Hybrid function. Here, 3 unique models are validated using
    the best performing UNET_AE (pretrained) from trial_1 and the best
    performing RNN/GRU/LSTM (pretrained) from trials_2 - trial_4.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 5: Hybrid MD RNN UNET')
    _train_loaders, _valid_loaders = get_Hybrid_loaders(
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
    for i in range(2, 3):
        _p = mp.Process(
            target=trial_5_Hybrid,
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


def trial_5_error_timeline():
    """The trial_5_error_timeline function creates an error timeline for each
    Hybrid MD RNN UNET model and each validation dataset. The resulting error
    timelines are plotted via the compareErrorTimeline_np function.

    Args:
        NONE

    Returns:
        NONE
    """
    print('Starting Trial 5: Hybrid MD RNN UNET Error Timeline')

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
        f'{_directory}1_UNET_AE/{_unet_name}'))

    _model_rnn_1 = RNN(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=1,
        device=device
    )
    _model_rnn_1.load_state_dict(torch.load(
        f'{_directory}2_RNN/{model_name_1}'))
    _models.append(_model_rnn_1)

    _model_rnn_2 = GRU(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_2.load_state_dict(torch.load(
        f'{_directory}3_GRU/{model_name_2}'))
    _models.append(_model_rnn_2)

    _model_rnn_3 = LSTM(
        input_size=256,
        hidden_size=256,
        seq_size=25,
        num_layers=2,
        device=device
    )
    _model_rnn_3.load_state_dict(torch.load(
        f'{_directory}4_LSTM/{model_name_3}'))
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
            _error_timeline = error_timeline(
                loader=_loader,
                model=_model,
                criterion=_criterion
            )
            _error_timelines[i].append(_error_timeline)

    compareErrorTimeline_np(
        l_of_l_losses=_error_timelines,
        l_of_l_labels=_model_identifiers,
        l_of_titles=_dataset_identifiers,
        file_prefix=f'{_directory}/5_Hybrid/',
        file_name='Hybrid_Models'
    )
    return


if __name__ == "__main__":
    trial_5_Hybrid_mp()
    pass
