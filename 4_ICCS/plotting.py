"""plotting

This script contains all plotting functionalities used for the entirety of this
paper. In particular:

plot_flow_profile

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    t, c, d, h, w = dataset.shape
    mid = int(h/2)
    avg_ux = []
    avg_uy = []
    avg_uz = []

    for dt in range(t):
        avg_ux.append(dataset[dt, 0, mid, :, mid].mean())
        avg_uy.append(dataset[dt, 1, mid, :, mid].mean())
        avg_uz.append(dataset[dt, 2, mid, :, mid].mean())

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharex=True, constrained_layout=True)

    fig.suptitle('Average Velocity Components vs Time', fontsize=10)

    ax1.set_xlabel("t")
    ax1.set_ylabel("$u_x$")
    ax1.grid(axis='y', alpha=0.3)

    ax2.set_xlabel("t")
    ax2.set_ylabel("$u_y$")
    ax2.grid(axis='y', alpha=0.3)

    ax3.set_xlabel("t")
    ax3.set_ylabel("$u_z$")
    ax2.grid(axis='y', alpha=0.3)

    ax1.plot(avg_ux, linewidth=2)
    ax2.plot(avg_uy, linewidth=2)
    ax3.plot(avg_uz, linewidth=2)

    fig.savefig(f'Plot_flow_profile_{dataset_name}.png')
    plt.close()
