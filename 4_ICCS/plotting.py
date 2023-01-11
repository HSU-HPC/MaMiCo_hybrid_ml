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
