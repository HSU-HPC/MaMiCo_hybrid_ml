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

    t, c, d, h, w = dataset.shape
    mid = int(h/2)
    avg_ux = []
    avg_uy = []
    avg_uz = []

    for t in range(250):
        avg_ux.append(dataset[t, 0, mid, :, mid].mean())
        avg_uy.append(dataset[t, 1, mid, :, mid].mean())
        avg_uz.append(dataset[t, 2, mid, :, mid].mean())

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharex=True, constrained_layout=True)

    fig.suptitle('Average Velocity Components vs Time', fontsize=10)

    ax1.set_xlabel("t")
    ax1.set_ylabel("$u_x$")

    ax2.set_xlabel("t")
    ax2.set_ylabel("$u_y$")

    ax3.set_xlabel("t")
    ax3.set_ylabel("$u_z$")

    ax1.plot(avg_ux)
    ax2.plot(avg_uy)
    ax3.plot(avg_uz)

    fig.savefig(f'Plot_flow_profile_{dataset_name}.png')
    plt.close()
