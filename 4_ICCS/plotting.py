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


def plot_flow_profile(dataset, dataset_name):
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
    dataset_name = dataset_name.replace('.csv', '')
    dataset_name = dataset_name.replace('02_clean/', '')
    t, c, d, h, w = dataset.shape
    mid = int(h/2)
    avg_ux = []
    avg_uy = []
    avg_uz = []

    for dt in range(t):
        avg_ux.append(dataset[dt, 0, :, :, :].mean())
        avg_uy.append(dataset[dt, 1, :, :, :].mean())
        avg_uz.append(dataset[dt, 2, :, :, :].mean())

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharex=True, constrained_layout=True)

    fig.suptitle(
        f'Average Velocity Components vs Time: {dataset_name}', fontsize=10)

    ax1.set_xlabel("t")
    ax1.set_ylabel("domain averaged $u_x$")
    ax1.grid(axis='y', alpha=0.3)

    ax2.set_xlabel("t")
    ax2.set_ylabel("domain averaged $u_y$")
    ax2.grid(axis='y', alpha=0.3)

    ax3.set_xlabel("t")
    ax3.set_ylabel("domain averaged $u_z$")
    ax3.grid(axis='y', alpha=0.3)

    ax1.plot(avg_ux, linewidth=0.3)
    ax2.plot(avg_uy, linewidth=0.3)
    ax3.plot(avg_uz, linewidth=0.3)

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


def plotPredVsTargKVS(input_1, input_2='void', file_prefix=0, file_name=0):
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
    p_avg_y = np.mean(input_1[:, 1, :, :, :], axis=(1, 2, 3))
    t_avg_y = np.mean(input_2[:, 1, :, :, :], axis=(1, 2, 3))
    p_avg_z = np.mean(input_1[:, 2, :, :, :], axis=(1, 2, 3))
    t_avg_z = np.mean(input_2[:, 2, :, :, :], axis=(1, 2, 3))

    p_loc_x = np.mean(input_1[:, 0, mid, mid, :], axis=(1))
    print('p_loc_x.shape: ', p_loc_x.shape)
    p_loc_y = np.mean(input_1[:, 1, mid, mid, :], axis=(1))
    p_loc_z = np.mean(input_1[:, 2, mid, mid, :], axis=(1))
    t_loc_x = np.mean(input_2[:, 0, mid, mid, :], axis=(1))
    t_loc_y = np.mean(input_2[:, 1, mid, mid, :], axis=(1))
    t_loc_z = np.mean(input_2[:, 2, mid, mid, :], axis=(1))

    print('std array shape: ', t_std_z.shape)
    print('avg array shape: ', t_avg_z.shape)

    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
    axs[0].plot(t_axis, p_avg_x, linewidth=0.5, label='Prediction')
    axs[0].fill_between(t_axis, p_avg_x-p_std_x, p_avg_x
                        + p_std_x, alpha=0.2, label='Prediction')
    axs[0].plot(t_axis, t_avg_x, linewidth=0.5, label='Target')
    axs[0].fill_between(t_axis, t_avg_x-t_std_x, t_avg_x
                        + t_std_x, alpha=0.2, label='Target')
    axs[0].scatter(t_axis, p_loc_x, s=0.05, label='[P] Central Cell')
    axs[0].scatter(t_axis, t_loc_x, s=0.05, label='[T] Central Cell')
    axs[0].set_ylabel('Averaged $u_x$')
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].plot(t_axis, p_avg_y, linewidth=0.5, label='Prediction')
    axs[1].fill_between(t_axis, p_avg_y-p_std_y, p_avg_y
                        + p_std_y, alpha=0.2, label='Prediction')
    axs[1].plot(t_axis, t_avg_y, linewidth=0.5, label='Target')
    axs[1].fill_between(t_axis, t_avg_y-t_std_y, t_avg_y
                        + t_std_y, alpha=0.2, label='Target')
    axs[1].scatter(t_axis, p_loc_y, s=0.05, label='[P] Central Cell')
    axs[1].scatter(t_axis, t_loc_y, s=0.05, label='[T] Central Cell')
    axs[1].set_ylabel('Averaged $u_y$')
    axs[1].grid(axis='y', alpha=0.3)

    axs[2].plot(t_axis, p_avg_z, linewidth=0.5, label='Prediction')
    axs[2].fill_between(t_axis, p_avg_z-p_std_z, p_avg_z
                        + p_std_z, alpha=0.2, label='Prediction')
    axs[2].plot(t_axis, t_avg_z, linewidth=0.5, label='Target')
    axs[2].fill_between(t_axis, t_avg_z-t_std_z, t_avg_z
                        + t_std_z, alpha=0.2, label='Target')
    axs[2].scatter(t_axis, p_loc_z, linewidth=0.05, label='[P] Central Cell')
    axs[2].scatter(t_axis, t_loc_z, linewidth=0.05, label='[T] Central Cell')
    axs[2].set_ylabel('Averaged $u_z$')
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


if __name__ == "__main__":
    pred = np.random.rand(1000, 3, 24, 24, 24) + 1
    targ = np.random.rand(1000, 3, 24, 24, 24) + 2.2
    plotPredVsTargKVS(input_1=pred, input_2=targ, file_name='test')
