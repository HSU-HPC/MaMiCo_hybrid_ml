# from ptflops import get_model_complexity_info
# from torchsummary import summary
# import torchvision.transforms.functional as TF
import torch.nn as nn
import torch
import numpy as np
from torchvision import models
from torchsummary import summary
# from ptflops import get_model_complexity_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


def tensor_FIFO_pipe(tensor, x, device):
    """The tensor_FIFO_pipe function acts as a first-in-first-out updater and
    is required for the hybrid models. It takes a tensor 'tensor' containing
    information from previous timesteps and concatenates new information 'x'
    to the front fo 'tensor'. In a FIFO manner, it returns all but the last
    element of 'tensor'.

    Args:
        tensor:
          Object of PyTorch-type tensor containing information from previous
          timesteps.
        x:
          Object of PyTorch-type tensor containing information from current
          timestep.
    Return:
        result:
          Object of PyTorch-type tensor upated in a FIFO manner.
    """
    result = torch.cat((tensor[1:].to(device), x.to(device)))
    return result


class DoubleConv(nn.Module):
    """The DoubleConv class is created as a building block for the U-Net
    inspired convolutional neural networks used in this project.

    This building block acts as a modular sequence of three dimensional
    convolutional operators followed by a specified activation function. It is
    modular in that it can be tailored to a desired number of input and output
    channels as well as a specific activation function.

    This implementation is inspired by an approach from Aladdin Persson
    (YouTube, GitHub: https://github.com/aladdinpersson)

    Attributes:
        in_channels:
          Object of integer type describing number of channels in the input data.
        out_channels:
          Object of integer type describing number of channels in the output data.
        activation:
          Object of PyTorch type torch.nn containing an activation function
    """

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1,
                      bias=False),
            # PARAMETERS:
            # 3: kernel_size
            # 1: stride
            # 1: padding -> same padding
            activation,
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class AE(nn.Module):
    """The AE class aims at implementing a strictly convolutional autoencoder
    very comparable to the above implemented U-Net autoencoder.

    Attributes:
        in_channels:
          Object of integer type describing number of channels in the input data.
        out_channels:
          Object of integer type describing number of channels in the output data.
        features:
          Object of type List containing integers that correspond to the number
          of kernels applied per convolutional
        activation:
          Object of PyTorch type torch.nn containing an activation function
    """

    def __init__(self, device, in_channels=3, out_channels=3, features=[4, 6, 8, 10], activation=nn.ReLU(inplace=True)):
        super(AE, self).__init__()
        self.device = device

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.helper_down = nn.Conv3d(
            in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.helper_up_1 = nn.ConvTranspose3d(
            in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0, bias=False)
        self.helper_up_2 = nn.Conv3d(
            in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        # Down part of AE
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, activation))
            in_channels = feature

        # Up part of AE
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature, feature, activation))

        # This is the "deepest" part.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, activation)
        print('Model initialized: Autoencoder.')

    def forward(self, x, y=0, skip_connections=0):
        """The forward method acts as a quai-overloaded method in that depending
        on the passed flag 'y', the forward method begins and returns different
        values. This is necessary to later feed time-series latent space predictions
        back into the model.

        Args:
            x:
              Object of PyTorch-type tensor containing the information of a timestep.
            y:
              Object of string type acting as a flag to choose desired forward method.
            skip_connections:
              Object of type list containing objects of PyTorch-type tensor that
              contain the U-Net unique skip_connections for later concatenation.
        Return:
            result:
              Object of PyTorch-type tensor returning the autoencoded result.
        """
        skip_connections = []
        if y == 0 or y == 'get_bottleneck':
            # The following for-loop describes the entire (left) contracting side,
            for down in self.downs:
                x = down(x)
                x = self.pool(x)

            # This is the bottleneck
            x = self.helper_down(x)
            x = self.activation(x)
            x = self.bottleneck(x)
            x = self.activation(x)

            if y == 'get_bottleneck':
                return x, skip_connections

            x = self.helper_up_1(x)
            x = self.activation(x)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)

            x = self.helper_up_2(x)

            return x

        if y == 'get_MD_output':
            x = self.helper_up_1(x)
            x = self.activation(x)

            # The following for-loop describes the entire (right) expanding side.
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                x = self.ups[idx+1](x)

            x = self.helper_up_2(x)
            return x


class RNN(nn.Module):
    """The RNN class aims at implementing an adapted RNN to be used in the hybrid
    model for time-series prediction of unrolled latent spaces.

    input.shape  = (batch_size, num_seq, input_size)
    output.shape = (batch_size, 1, input_size)

    Attributes:
        input_size:
          Object of integer type describing the number of features in a single
          element of the time-series sequence.
        hidden_size:
          Object of integer type describing the number of features in the hidden
          state, i.e. the number of features in the output.
        seq_size:
          Object of integer type describing the number of elements in the
          time-series sequence.
        num_layers:
          Object of integer type describing the number of stacked RNN units.
    """

    def __init__(self, input_size, hidden_size, seq_size, num_layers, device):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.num_layers = num_layers
        self.device = device
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size*self.seq_size, self.input_size)
        print('Model initialized: RNN.')

    def forward(self, x):
        # Set initial hidden states(for RNN, GRU, LSTM)
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = out.reshape(out.shape[0], -1)

        # Apply linear regressor to the last time step
        out = self.fc(out)
        return out


class GRU(nn.Module):
    """The GRU class aims at implementing an adapted GRU to be used in the hybrid
    model for time-series prediction of unrolled latent spaces.

    input.shape  = (batch_size, num_seq, input_size)
    output.shape = (batch_size, 1, input_size)

    Attributes:
        input_size:
          Object of integer type describing the number of features in a single
          element of the time-series sequence.
        hidden_size:
          Object of integer type describing the number of features in the hidden
          state, i.e. the number of features in the output.
        seq_size:
          Object of integer type describing the number of elements in the
          time-series sequence.
        num_layers:
          Object of integer type describing the number of stacked RNN units.
    """

    def __init__(self, input_size, hidden_size, seq_size, num_layers, device):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size*self.seq_size, self.input_size)
        print('Model initialized: GRU.')

    def forward(self, x):
        # Set initial hidden states(for RNN, GRU, LSTM)
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = out.reshape(out.shape[0], -1)

        # Apply linear regressor to the last time step
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    """The LSTM class aims at implementing an adapted LSTM to be used in the hybrid
    model for time-series prediction of unrolled latent spaces.

    input.shape  = (batch_size, num_seq, input_size)
    output.shape = (batch_size, 1, input_size)

    Attributes:
        input_size:
          Object of integer type describing the number of features in a single
          element of the time-series sequence.
        hidden_size:
          Object of integer type describing the number of features in the hidden
          state, i.e. the number of features in the output.
        seq_size:
          Object of integer type describing the number of elements in the
          time-series sequence.
        num_layers:
          Object of integer type describing the number of stacked RNN units.
    """

    def __init__(self, input_size, hidden_size, seq_size, num_layers, device):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size*self.seq_size, input_size)
        print('Model initialized: LSTM.')

    def forward(self, x):
        # Set initial hidden states(for RNN, GRU, LSTM)
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out.reshape(out.shape[0], -1)

        # Apply linear regressor to the last time step
        out = self.fc(out)
        return out


class Hybrid_MD_RNN_AE(nn.Module):
    """The Hybrid_MD_RNN_AE class aims at implementing the strictly convolutional
    autoencoder and RNN based hybrid model for time-series prediction of MD
    velocity distributions.

    input.shape  = (batch_size = 1, 3, 24, 24, 24)
    output.shape = (batch_size = 1, 3, 24, 24, 24)

    Attributes:
        AE_Model:
          Object of type torch.nn.Module containing a trained strictly convolutional
          autoencoder model
        RNN_Model:
          Object of type torch.nn.Module containing a trained RNN model
        seq_length:
          Object of integer type describing the number of elements to be
          considered in the time-series sequence of latent spaces.
    """

    def __init__(self, device, AE_Model, RNN_Model, seq_length):
        super(Hybrid_MD_RNN_AE, self).__init__()
        self.device = device
        self.AE = AE_Model.eval()
        self.rnn = RNN_Model.eval()
        self.seq_length = seq_length
        self.sequence = torch.zeros(self.seq_length, 256)
        print('Model initialized: Hybrid_MD_RNN_AE')

    def forward(self, x):

        x, skip_connections = self.AE(x, y='get_bottleneck')

        x_shape = x.shape
        self.sequence = tensor_FIFO_pipe(
            tensor=self.sequence,
            x=torch.reshape(x, (1, 256)),
            device=self.device).to(self.device)

        interim = torch.reshape(self.sequence, (1, self.seq_length, 256))
        x = self.rnn(interim)

        x = torch.reshape(x, x_shape)

        x = self.AE(x, y='get_MD_output', skip_connections=skip_connections)
        return x


def resetPipeline(model):
    """The resetPipeline function is required to reset the model.sequence
    attribute that contains the latent spaces from previous timesteps. In
    particular, this is necessary to prepare the model for data from a new data
    distribution.

    Args:
        model:
          Object of PyTorch-type nn.Module containing one of the above defined
          hybrid models
    Return:
        none
    """
    shape = model.sequence.size()
    model.sequence = torch.zeros(shape)
    return


def test_forward_overloading():
    """The test_forward_overloading function aims to check UNET_AE's workaround
    forward overloading. This functionality is vital in order to train and save
    the model for hybrid use where a RNN is used to pass a unique bottleneck
    value. This function checks if the functionality yields identical tensors:
    x_check_1 = model(x)
    x_interim, skips = model(x, y=get_interim)
    x_check_2 = model(x_interim,y=get_output, skips=skips)

    Args:
        none
    Return:
        none
    """

    model = AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=nn.ReLU(inplace=True)
    )

    _x_in = torch.ones(1000, 3, 24, 24, 24)
    _x_out_1 = model(x=_x_in)

    _x_bottleneck, _skips = model(x=_x_in, y='get_bottleneck')

    _x_out_2 = model(x=_x_bottleneck, y='get_MD_output',
                     skip_connections=_skips)

    if torch.equal(_x_out_1, _x_out_2):
        print('Tensors are equal.')
    else:
        print('Tensors are not equal.')


def calculateFLOPS_ConvAE(c, d, h, w, features):
    """The calculateFLOPS_ConvAE function is used to calculate the number of
    FLOPs and MACCs required to perform a predicition. In particular it considers
    the following relationships:
    - Convolutional Layers
      MACCs: number of kernels x kernel shape x output shape
      FLOPs: 2 * MACCs
    - Pooling Layers
      FLOPs: channels x (depth/stride) x (height/stride) x (width/stride)
    - Activations
      FLOPs: channels x depth x height x width

    Args:
        c:
          Number of channels in the input volume
        d:
          Depth of the input volume
        h:
          Height of the input volume
        w:
          Width of the input volume
        features:

    Returns:
        NONE:
          This function returns the number of MACCs and FLOPs required to
          perform a single prediction.
    """
    _FLOPs = 0
    _MACCs = 0
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    def get_MACCs_Conv(i_channels, n_kernels, k_shape, o_shape):
        k_vol = k_shape**3
        o_vol = k_shape**3
        return n_kernels * i_channels * k_vol * o_vol

    def get_FLOPs_Pool(c, d, h, w):
        s = 2
        return c * d/s * h/s * w/s

    def get_FLOPs_Activ(c, d, h, w):
        return c * d * h * w

    for down in features:
        #DoubleConvs
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=down, k_shape=3, o_shape=h)
        c = down
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _FLOPs += get_FLOPs_Pool(c, d, h, w)
        d = d/2
        h = h/2
        w = w/2
        print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_down
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=c, k_shape=2, o_shape=h)
    d = d - 1
    h = h - 1
    w = w - 1
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # Bottleneck DoubleConv
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=2*c, k_shape=3, o_shape=h)
    c = 2 * c
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    _MACCs += get_MACCs_Conv(i_channels=c,
                             n_kernels=down, k_shape=3, o_shape=h)
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_up_1
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=c, k_shape=2, o_shape=h)
    d = d + 1
    h = h + 1
    w = w + 1
    _FLOPs += get_FLOPs_Activ(c, d, h, w)
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    for up in reversed(features):
        # First Up-Convolution
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=up, k_shape=2, o_shape=h*2)
        c = up
        d = d*2
        h = h*2
        w = w*2
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        _MACCs += get_MACCs_Conv(i_channels=c,
                                 n_kernels=c, k_shape=3, o_shape=h)
        _FLOPs += get_FLOPs_Activ(c, d, h, w)
        print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    # helper_up_2
    _MACCs += get_MACCs_Conv(i_channels=c, n_kernels=3, k_shape=3, o_shape=h)
    c = 3
    print(f'Current Dimensions: {int(c)}x{int(d)}x{int(h)}x{int(w)}')

    print(f'Total number of  MACCs: {_MACCs}')
    print(f'Total number of  FLOPs: {_FLOPs + 2*_MACCs}')

    print(f'Total number of GFLOPs: {(_FLOPs + 2*_MACCs)/(10**9)}')


if __name__ == "__main__":
    pass
