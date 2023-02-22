"""plotting

This script contains all plotting functionalities used for the entirety of this
paper. In particular:

plot_flow_profile

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import csv2dataset_mp


def getColor(c, N, idx):
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def plot_flow_profile(file_name, dataset_md, dataset_lbm=None):
    """The plot_flow_profile function visualizes datasets via 2D flow profiles
    and as such is used to validate proper simulation/prediction. For our
    purposes, we create 3 subplots to validate couette and KVS based simulations.
        x-axis: time steps
        y-axis: averaged velocity component (u_x, u_y, u_z) as determined over
        the column of central cells in y-direction (e.g. [t, 2, 12, :, 12]).

        directory:
          Object of string type containing the path working directory (pwd) of
          the dataset to be visualized. This is also the pwd of the plots.
        file_name:
          Object of string type containing the name of the csv file to be
          visualized.

    Returns:
        NONE:
          This function does not have a return value. Instead it generates the
          aforementioned meaningful plots.
    """
    dataset_name = file_name.replace('.csv', '')
    dataset_name = dataset_name.replace('02_clean/', '')
    dataset_name = dataset_name.replace('/', '')
    t, c, d, h, w = dataset_md.shape
    # mid = int(h/2)t_max = 1000
    t_max = 1000
    t_axis = np.arange(1, t_max+1)
    lbm_loc_ux = dataset_lbm[:, 1]
    lbm_loc_uy = dataset_lbm[:, 2]
    avg_ux = []
    avg_uy = []

    for dt in range(t):
        avg_ux.append(dataset_md[dt, 0, :, :, :].mean())
        avg_uy.append(dataset_md[dt, 1, :, :, :].mean())

    fig, axs = plt.subplots(
        2, sharex=True, constrained_layout=True)

    fig.suptitle(
        f'Average Velocity Components vs Time: {dataset_name}', fontsize=10)

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("$u_x$")
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].set_xlabel("t")
    axs[1].set_ylabel("$u_y$")
    axs[1].grid(axis='y', alpha=0.3)

    axs[0].plot(t_axis, avg_ux, linewidth=0.3, label='md avg u_x')
    axs[1].plot(t_axis, avg_uy, linewidth=0.3, label='md avg u_y')

    if dataset_lbm is not None:
        axs[0].plot(t_axis, lbm_loc_ux, linewidth=0.3, label='lbm local u_x')
        axs[1].plot(t_axis, lbm_loc_uy, linewidth=0.3, label='lbm local u_y')

    axs[0].legend(ncol=1, fontsize=9)
    axs[1].legend(ncol=1, fontsize=9)
    fig.savefig(f'plots/Plot_flow_profile_{dataset_name}.png')
    plt.close()


def compareLossVsValid(loss_files, loss_labels, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    losses = csv2dataset_mp(loss_files)

    loss_list = []
    for loss in losses:
        loss_list.append(loss)

    num_epoch = (loss_list[0]).shape[0]

    x_axis = range(1, (num_epoch+1), 1)

    fig, (ax1) = plt.subplots(1, constrained_layout=True)
    for idx in range(int(len(loss_list)/2)):
        ax1.plot(x_axis, loss_list[2*idx], color=getColor(c='tab20',
                 N=12, idx=idx), label=loss_labels[2*idx])
        ax1.plot(x_axis, loss_list[2*idx+1], color=getColor(c='tab20',
                 N=12, idx=idx), linestyle='--', label=loss_labels[2*idx+1])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(ncol=2, fontsize=9)
    fig.set_size_inches(7, 2.5)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Loss_vs_Valid_{file_name}.svg')

    plt.close()


def plotPredVsTargKVS(input_1, input_2='void', input_3='void', file_prefix=0, file_name=0):
    """The plotPredVsTargKVS function aims to graphically compare model
    performance via plotting domain-wise averaged predicted and target
    velocities vs time. The standard deviations are additionally included for
    better comparison.

    Args:
        input_1:
          Object of type PyTorch-Tensor containing the predicted dataset
        input_2:
          Object of type PyTorch-Tensor containing the target dataset
        file_prefix:
          Object of type string containing
        file_name:
          Object of type string containing

    Returns:
        NONE:
          This function saves the graphical comparison to file.
    """

    if input_2 == 'void':
        print('Invalid input_2.')

    t, c, x, y, z = input_1.shape

    #  mid = y//2
    mid = 6
    t_max = t
    t_axis = np.arange(1, t_max+1)

    p_std_x = np.std(input_1[:, 0, :, :, :], axis=(1, 2, 3))
    t_std_x = np.std(input_2[:, 0, :, :, :], axis=(1, 2, 3))
    p_std_y = np.std(input_1[:, 1, :, :, :], axis=(1, 2, 3))
    t_std_y = np.std(input_2[:, 1, :, :, :], axis=(1, 2, 3))
    p_std_z = np.std(input_1[:, 2, :, :, :], axis=(1, 2, 3))
    t_std_z = np.std(input_2[:, 2, :, :, :], axis=(1, 2, 3))

    p_avg_x = np.mean(input_1[:, 0, :, :, :], axis=(1, 2, 3))
    t_avg_x = np.mean(input_2[:, 0, :, :, :], axis=(1, 2, 3))
    p_avg_y = np.mean(input_1[:, 1, :, :, :], axis=(1, 2, 3))
    t_avg_y = np.mean(input_2[:, 1, :, :, :], axis=(1, 2, 3))
    p_avg_z = np.mean(input_1[:, 2, :, :, :], axis=(1, 2, 3))
    t_avg_z = np.mean(input_2[:, 2, :, :, :], axis=(1, 2, 3))

    my_s = 0.25

    p_loc_x = input_1[:, 0, mid, mid, mid]
    p_loc_y = input_1[:, 1, mid, mid, mid]
    p_loc_z = input_1[:, 2, mid, mid, mid]
    t_loc_x = input_2[:, 0, mid, mid, mid]
    t_loc_y = input_2[:, 1, mid, mid, mid]
    lbm_loc_y = input_3[:, 2]
    t_loc_z = input_2[:, 2, mid, mid, mid]

    print('std array shape: ', t_std_z.shape)
    print('avg array shape: ', t_avg_z.shape)

    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
    '''
    axs[0].plot(t_axis, p_avg_x, linewidth=0.5, label='Prediction')
    axs[0].fill_between(t_axis, p_avg_x-p_std_x, p_avg_x
                        + p_std_x, alpha=0.2, label='Prediction')
    axs[0].plot(t_axis, t_avg_x, linewidth=0.5, label='Target')
    axs[0].fill_between(t_axis, t_avg_x-t_std_x, t_avg_x
                        + t_std_x, alpha=0.2, label='Target')
    '''
    axs[0].scatter(t_axis, p_loc_x, s=my_s, label='[P] Central Cell')
    axs[0].scatter(t_axis, t_loc_x, s=my_s, label='[T] Central Cell')
    axs[0].set_ylabel('Local $u_x$')
    axs[0].grid(axis='y', alpha=0.3)

    '''
    axs[1].plot(t_axis, p_avg_y, linewidth=0.5, label='Prediction')
    axs[1].fill_between(t_axis, p_avg_y-p_std_y, p_avg_y
                        + p_std_y, alpha=0.2, label='Prediction')
    axs[1].plot(t_axis, t_avg_y, linewidth=0.5, label='Target')
    axs[1].fill_between(t_axis, t_avg_y-t_std_y, t_avg_y
                        + t_std_y, alpha=0.2, label='Target')
    '''
    axs[1].scatter(t_axis, p_loc_y, s=my_s, label='[P] Central Cell')
    axs[1].scatter(t_axis, t_loc_y, s=my_s, label='[T] Central Cell')
    axs[1].set_ylabel('Local $u_y$')
    axs[1].grid(axis='y', alpha=0.3)

    '''
    axs[2].plot(t_axis, p_avg_z, linewidth=0.5, label='Prediction')
    axs[2].fill_between(t_axis, p_avg_z-p_std_z, p_avg_z
                        + p_std_z, alpha=0.2, label='Prediction')
    axs[2].plot(t_axis, t_avg_z, linewidth=0.5, label='Target')
    axs[2].fill_between(t_axis, t_avg_z-t_std_z, t_avg_z
                        + t_std_z, alpha=0.2, label='Target')
    '''
    axs[2].scatter(t_axis, p_loc_y, s=my_s, label='[P] Central Cell')
    axs[2].scatter(t_axis, lbm_loc_y, s=my_s, label='[T] lbm')
    axs[2].set_ylabel('Local $u_y$')
    axs[2].grid(axis='y', alpha=0.3)

    axs[2].set_xlabel('Timestep')
    axs[2].legend(ncol=3, fontsize=9)

    fig.set_size_inches(6, 4)
    if file_name != 0:
        fig.savefig(
            f'{file_prefix}Plot_PredVsTarg_KVS_{file_name}.svg')
    #plt.show()
    plt.close()
    pass


def plotPredVsTargKVS_new(input_1, input_2='void', file_prefix=0, file_name=0):
    """The plotPredVsTargKVS function aims to graphically compare model
    performance via plotting domain-wise averaged predicted and target
    velocities vs time. The standard deviations are additionally included for
    better comparison.

    Args:
        input_1:
          Object of type PyTorch-Tensor containing the predicted dataset
        input_2:
          Object of type PyTorch-Tensor containing the target dataset
        file_prefix:
          Object of type string containing
        file_name:
          Object of type string containing

    Returns:
        NONE:
          This function saves the graphical comparison to file.
    """

    if input_2 == 'void':
        print('Invalid input_2.')

    t, c, x, y, z = input_1.shape
    mid = y//2
    t_max = t
    t_axis = np.arange(1, t_max+1)

    p_std_x = np.std(input_1[:, 0, :, :, :], axis=(1, 2, 3))
    t_std_x = np.std(input_2[:, 0, :, :, :], axis=(1, 2, 3))
    p_std_y = np.std(input_1[:, 1, :, :, :], axis=(1, 2, 3))
    t_std_y = np.std(input_2[:, 1, :, :, :], axis=(1, 2, 3))
    p_std_z = np.std(input_1[:, 2, :, :, :], axis=(1, 2, 3))
    t_std_z = np.std(input_2[:, 2, :, :, :], axis=(1, 2, 3))

    p_avg_x = np.mean(input_1[:, 0, :, :, :], axis=(1, 2, 3))
    t_avg_x = np.mean(input_2[:, 0, :, :, :], axis=(1, 2, 3))
    p_avg_x_NE = np.mean(input_1[:, 0, mid:, mid:, :], axis=(1, 2, 3))
    t_avg_x_NE = np.mean(input_2[:, 0, mid:, mid:, :], axis=(1, 2, 3))
    p_avg_x_NW = np.mean(input_1[:, 0, :mid, mid:, :], axis=(1, 2, 3))
    t_avg_x_NW = np.mean(input_2[:, 0, :mid, mid:, :], axis=(1, 2, 3))
    p_avg_x_SE = np.mean(input_1[:, 0, mid:, :mid, :], axis=(1, 2, 3))
    t_avg_x_SE = np.mean(input_2[:, 0, mid:, :mid, :], axis=(1, 2, 3))
    p_avg_x_SW = np.mean(input_1[:, 0, :mid, :mid, :], axis=(1, 2, 3))
    t_avg_x_SW = np.mean(input_2[:, 0, :mid, :mid, :], axis=(1, 2, 3))

    p_avg_y = np.mean(input_1[:, 1, :, :, :], axis=(1, 2, 3))
    t_avg_y = np.mean(input_2[:, 1, :, :, :], axis=(1, 2, 3))
    p_avg_y_NE = np.mean(input_1[:, 1, mid:, mid:, :], axis=(1, 2, 3))
    t_avg_y_NE = np.mean(input_2[:, 1, mid:, mid:, :], axis=(1, 2, 3))
    p_avg_y_NW = np.mean(input_1[:, 1, :mid, mid:, :], axis=(1, 2, 3))
    t_avg_y_NW = np.mean(input_2[:, 1, :mid, mid:, :], axis=(1, 2, 3))
    p_avg_y_SE = np.mean(input_1[:, 1, mid:, :mid, :], axis=(1, 2, 3))
    t_avg_y_SE = np.mean(input_2[:, 1, mid:, :mid, :], axis=(1, 2, 3))
    p_avg_y_SW = np.mean(input_1[:, 1, :mid, :mid, :], axis=(1, 2, 3))
    t_avg_y_SW = np.mean(input_2[:, 1, :mid, :mid, :], axis=(1, 2, 3))

    p_avg_z = np.mean(input_1[:, 2, :, :, :], axis=(1, 2, 3))
    t_avg_z = np.mean(input_2[:, 2, :, :, :], axis=(1, 2, 3))
    p_avg_z_NE = np.mean(input_1[:, 2, mid:, mid:, :], axis=(1, 2, 3))
    t_avg_z_NE = np.mean(input_2[:, 2, mid:, mid:, :], axis=(1, 2, 3))
    p_avg_z_NW = np.mean(input_1[:, 2, :mid, mid:, :], axis=(1, 2, 3))
    t_avg_z_NW = np.mean(input_2[:, 2, :mid, mid:, :], axis=(1, 2, 3))
    p_avg_z_SE = np.mean(input_1[:, 2, mid:, :mid, :], axis=(1, 2, 3))
    t_avg_z_SE = np.mean(input_2[:, 2, mid:, :mid, :], axis=(1, 2, 3))
    p_avg_z_SW = np.mean(input_1[:, 2, :mid, :mid, :], axis=(1, 2, 3))
    t_avg_z_SW = np.mean(input_2[:, 2, :mid, :mid, :], axis=(1, 2, 3))

    p_loc_x = np.mean(input_1[:, 0, :, mid, mid], axis=(1))
    print('p_loc_x.shape: ', p_loc_x.shape)
    p_loc_y = np.mean(input_1[:, 1, :, mid, mid], axis=(1))
    p_loc_z = np.mean(input_1[:, 2, :, mid, mid], axis=(1))
    t_loc_x = np.mean(input_2[:, 0, :, mid, mid], axis=(1))
    t_loc_y = np.mean(input_2[:, 1, :, mid, mid], axis=(1))
    t_loc_z = np.mean(input_2[:, 2, :, mid, mid], axis=(1))

    print('std array shape: ', t_std_z.shape)
    print('avg array shape: ', t_avg_z.shape)

    fig, axs = plt.subplots(3, 2, sharex=True, constrained_layout=True)
    axs[0, 0].plot(t_axis, p_avg_x, linewidth=0.5, label='Prediction')
    axs[0, 0].fill_between(t_axis, p_avg_x-p_std_x, p_avg_x
                           + p_std_x, alpha=0.2, label='Prediction')
    axs[0, 0].plot(t_axis, t_avg_x, linewidth=0.5, label='Target')
    axs[0, 0].fill_between(t_axis, t_avg_x-t_std_x, t_avg_x
                           + t_std_x, alpha=0.2, label='Target')
    axs[0, 0].set_ylabel('Averaged $u_x$')
    axs[0, 0].grid(axis='y', alpha=0.3)

    axs[0, 1].plot(t_axis, p_avg_x_NE, color='green',
                   linewidth=0.5, label='NE')
    axs[0, 1].plot(t_axis, p_avg_x_NW, color='blue', linewidth=0.5, label='NW')
    axs[0, 1].plot(t_axis, p_avg_x_SE, color='magenta',
                   linewidth=0.5, label='SE')
    axs[0, 1].plot(t_axis, p_avg_x_SW, color='red', linewidth=0.5, label='SW')
    axs[0, 1].plot(t_axis, t_avg_x_NE, linestyle='dotted',
                   color='green', linewidth=0.5, label='NE')
    axs[0, 1].plot(t_axis, t_avg_x_NW, linestyle='dotted',
                   color='blue', linewidth=0.5, label='NW')
    axs[0, 1].plot(t_axis, t_avg_x_SE, linestyle='dotted',
                   color='magenta', linewidth=0.5, label='SE')
    axs[0, 1].plot(t_axis, t_avg_x_SW, linestyle='dotted',
                   color='red', linewidth=0.5, label='SW')
    axs[0, 1].set_ylabel('Averaged $u_x$')
    axs[0, 1].grid(axis='y', alpha=0.3)

    axs[1, 0].plot(t_axis, p_avg_y, linewidth=0.5, label='Prediction')
    axs[1, 0].fill_between(t_axis, p_avg_y-p_std_y, p_avg_y
                           + p_std_y, alpha=0.2, label='Prediction')
    axs[1, 0].plot(t_axis, t_avg_y, linewidth=0.5, label='Target')
    axs[1, 0].fill_between(t_axis, t_avg_y-t_std_y, t_avg_y
                           + t_std_y, alpha=0.2, label='Target')
    axs[1, 0].set_ylabel('Averaged $u_y$')
    axs[1, 0].grid(axis='y', alpha=0.3)

    axs[1, 1].plot(t_axis, p_avg_y_NE, color='green',
                   linewidth=0.5, label='NE')
    axs[1, 1].plot(t_axis, p_avg_y_NW, color='blue', linewidth=0.5, label='NW')
    axs[1, 1].plot(t_axis, p_avg_y_SE, color='magenta',
                   linewidth=0.5, label='SE')
    axs[1, 1].plot(t_axis, p_avg_y_SW, color='red', linewidth=0.5, label='SW')
    axs[1, 1].plot(t_axis, t_avg_y_NE, linestyle='dotted',
                   color='green', linewidth=0.5, label='NE')
    axs[1, 1].plot(t_axis, t_avg_y_NW, linestyle='dotted',
                   color='blue', linewidth=0.5, label='NW')
    axs[1, 1].plot(t_axis, t_avg_y_SE, linestyle='dotted',
                   color='magenta', linewidth=0.5, label='SE')
    axs[1, 1].plot(t_axis, t_avg_y_SW, linestyle='dotted',
                   color='red', linewidth=0.5, label='SW')
    axs[1, 1].set_ylabel('Averaged $u_y$')
    axs[1, 1].grid(axis='y', alpha=0.3)

    axs[2, 0].plot(t_axis, p_avg_z, linewidth=0.5, label='Prediction')
    axs[2, 0].fill_between(t_axis, p_avg_z-p_std_z, p_avg_z
                           + p_std_z, alpha=0.2, label='Prediction')
    axs[2, 0].plot(t_axis, t_avg_z, linewidth=0.5, label='Target')
    axs[2, 0].fill_between(t_axis, t_avg_z-t_std_z, t_avg_z
                           + t_std_z, alpha=0.2, label='Target')
    axs[2, 0].set_ylabel('Averaged $u_z$')
    axs[2, 0].grid(axis='y', alpha=0.3)

    axs[2, 0].set_xlabel('Timestep')

    axs[2, 1].plot(t_axis, p_avg_z_NE, color='green',
                   linewidth=0.5, label='NE')
    axs[2, 1].plot(t_axis, p_avg_z_NW, color='blue', linewidth=0.5, label='NW')
    axs[2, 1].plot(t_axis, p_avg_z_SE, color='magenta',
                   linewidth=0.5, label='SE')
    axs[2, 1].plot(t_axis, p_avg_z_SW, color='red', linewidth=0.5, label='SW')
    axs[2, 1].plot(t_axis, t_avg_z_NE, linestyle='dotted',
                   color='green', linewidth=0.5, label='NE')
    axs[2, 1].plot(t_axis, t_avg_z_NW, linestyle='dotted',
                   color='blue', linewidth=0.5, label='NW')
    axs[2, 1].plot(t_axis, t_avg_z_SE, linestyle='dotted',
                   color='magenta', linewidth=0.5, label='SE')
    axs[2, 1].plot(t_axis, t_avg_z_SW, linestyle='dotted',
                   color='red', linewidth=0.5, label='SW')
    axs[2, 1].set_ylabel('Averaged $u_z$')
    axs[2, 1].grid(axis='y', alpha=0.3)

    axs[2, 0].legend(ncol=3, fontsize=9)
    axs[2, 1].legend(ncol=4, fontsize=9)

    fig.set_size_inches(12, 4)
    if file_name != 0:
        fig.savefig(
            f'{file_prefix}Plot_PredVsTarg_KVS_{file_name}.svg')
    #plt.show()
    plt.close()
    pass


def plotlbm(lbm_input, file_directory, file_id):
    """The plotPredVsTargKVS function aims to graphically compare model
    performance via plotting domain-wise averaged predicted and target
    velocities vs time. The standard deviations are additionally included for
    better comparison.

    Args:
        input_1:
          Object of type PyTorch-Tensor containing the predicted dataset
        input_2:
          Object of type PyTorch-Tensor containing the target dataset
        file_prefix:
          Object of type string containing
        file_name:
          Object of type string containing

    Returns:
        NONE:
          This function saves the graphical comparison to file.
    """
    t_max = 1000
    t_axis = np.arange(1, t_max+1)
    lbm_loc_ux = lbm_input[:, 1]
    lbm_loc_uy = lbm_input[:, 2]

    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)

    axs[0].plot(t_axis, lbm_loc_ux, label='lbm local u_x')
    axs[0].set_ylabel('Local $u_x$')
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].plot(t_axis, lbm_loc_uy, label='lbm local u_y')
    axs[1].set_ylabel('Local $u_y$')
    axs[1].grid(axis='y', alpha=0.3)

    fig.set_size_inches(6, 3)
    if file_id != 0:
        fig.savefig(
            f'{file_directory}quick_and_dirty_verification_{file_id}.svg')
    plt.close()
    pass


if __name__ == "__main__":
    _lbm = np.loadtxt('dataset_mlready/kvs_20000_NE_lbm.csv', delimiter=";")
    _lbm = _lbm.reshape(1000, 3)
    _file_directory = 'plots/'
    _file_id = 'kvs_20000_NE'

    plotlbm(_lbm, _file_directory, _file_id)
