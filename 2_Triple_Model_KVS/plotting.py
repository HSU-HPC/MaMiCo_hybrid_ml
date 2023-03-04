"""plotting

This script contains all plotting functionalities used for the entirety of this
paper. In particular:

plot_flow_profile
plot_flow_profile_std

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


FONT = font_manager.FontProperties(weight='bold', size=10)


def plot_flow_profile(np_datasets, dataset_legends, save2file, unique_id=None):
    """The plot_flow_profile function visualizes datasets via 2D flow profiles
    and as such is used to validate proper simulation/prediction. For our
    purposes, we create 3 subplots to validate couette and KVS based simulations.
        x-axis: time steps
        y-axis: local velocity component (u_x, u_y, u_z) for a specific cell
        (e.g. [t, 2, 12, 12, 12]).
    Args:
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
    print('[plot_flow_profile()]')
    _save2file = save2file.replace('.csv', '')
    _save2file = _save2file.replace('01_clean_lbm/', '')
    _t_max = 850
    _t_axis = np.arange(1, _t_max+1)
    _n_datasets = len(np_datasets)

    for idx, dataset in enumerate(np_datasets):
        print(f'[{dataset_legends[idx]}] Dataset shape: ', dataset.shape)

    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)

    axs[0].set_ylabel(r'$\mathbf{u_x}$', fontproperties=FONT)
    axs[0].grid(axis='y', alpha=0.3)
    axs[0].set_ylim([1, 7])
    axs[0].set_yticks([2, 3, 4, 5, 6])
    axs[0].set_yticklabels([2, None, 4, None, 6], fontproperties=FONT)

    axs[1].set_ylabel(r'$\mathbf{u_y}$', fontproperties=FONT)
    axs[1].grid(axis='y', alpha=0.3)
    axs[1].set_ylim([-1.25, 1.25])
    axs[1].set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    axs[1].set_yticklabels([-1, None, 0, None, 1], fontproperties=FONT)

    axs[2].set_xlabel("t", fontproperties=FONT)
    axs[2].set_ylabel(r'$\mathbf{u_z}$', fontproperties=FONT)
    axs[2].grid(axis='y', alpha=0.3)
    axs[2].set_ylim([-1.25, 1.25])
    axs[2].set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    axs[2].set_yticklabels([-1, None, 0, None, 1], fontproperties=FONT)

    for idx, dataset in enumerate(np_datasets):
        mid = 12
        alpha = 1
        lw = 1.0
        if idx == 0:
            alpha = 0.7
            lw = 0.6
        axs[0].plot(_t_axis, dataset[-850:, 0, mid, mid, mid],
                    linewidth=lw, alpha=alpha, label=dataset_legends[idx])
        axs[1].plot(_t_axis, dataset[-850:, 1, mid, mid, mid],
                    linewidth=lw, alpha=alpha, label=dataset_legends[idx])
        axs[2].plot(_t_axis, dataset[-850:, 2, mid, mid, mid],
                    linewidth=lw, alpha=alpha, label=dataset_legends[idx])

    plt.xticks([0, 200, 400, 600, 800], fontproperties=FONT)
    fig.set_size_inches(7.85, 7)
    axs[2].legend(ncol=_n_datasets, prop=FONT, loc='lower left',
                  bbox_to_anchor=(0, -0.55), fancybox=True, shadow=False)

    if unique_id is not None:
        if unique_id == 0:
            plt.show()
            return
        else:
            fig.savefig(
                f'plots/Plot_loc_flow_profile_{save2file}{unique_id}.svg')
    else:
        fig.savefig(f'plots/Plot_loc_flow_profile_{save2file}.svg')
    # plt.show()
    plt.close()


def plot_flow_profile_std(np_datasets, dataset_legends, save2file, unique_id=None):
    """The plot_flow_profile function visualizes datasets via 2D flow profiles
    and as such is used to validate proper simulation/prediction. For our
    purposes, we create 3 subplots to validate couette and KVS based simulations.
        x-axis: time steps
        y-axis: local velocity component (u_x, u_y, u_z) for a specific cell
        (e.g. [t, 2, 12, 12, 12]).
    Args:
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
    print('[plot_flow_profile()]')
    _save2file = save2file.replace('.csv', '')
    _save2file = _save2file.replace('01_clean_lbm/', '')
    _t_max = 850
    _t_axis = np.arange(1, _t_max+1)
    _n_datasets = len(np_datasets)

    for idx, dataset in enumerate(np_datasets):
        print(f'[{dataset_legends[idx]}] Dataset shape: ', dataset.shape)

    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)

    axs[0].set_ylabel(r'$\mathbf{u_x}$', fontproperties=FONT)
    axs[0].grid(axis='y', alpha=0.3)
    axs[0].set_ylim([1.5, 6.5])
    axs[0].set_yticks([2, 3, 4, 5, 6])
    axs[0].set_yticklabels([2, None, 4, None, 6], fontproperties=FONT)

    axs[1].set_ylabel(r'$\mathbf{u_y}$', fontproperties=FONT)
    axs[1].grid(axis='y', alpha=0.3)
    axs[1].set_ylim([-0.55, 0.55])
    axs[1].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    axs[1].set_yticklabels([-0.5, None, 0, None, 0.5], fontproperties=FONT)

    axs[2].set_xlabel("t", fontproperties=FONT)
    axs[2].set_ylabel(r'$\mathbf{u_z}$', fontproperties=FONT)
    axs[2].grid(axis='y', alpha=0.3)
    axs[2].set_ylim([-0.55, 0.55])
    axs[2].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    axs[2].set_yticklabels([-0.5, None, 0, None, 0.5], fontproperties=FONT)

    for idx, dataset in enumerate(np_datasets):
        alpha = 0.4 * 2
        lw = 0.5 * 2
        mid = 12
        if idx == 0:
            alpha = 0.2 * 2
            lw = 0.3 * 2

        _d_std_x = np.std(dataset[-850:, 0, mid, mid, :], axis=(1))
        _d_std_y = np.std(dataset[-850:, 1, mid, mid, :], axis=(1))
        _d_std_z = np.std(dataset[-850:, 2, mid, mid, :], axis=(1))

        _d_avg_x = np.mean(dataset[-850:, 0, mid, mid, :], axis=(1))
        _d_avg_y = np.mean(dataset[-850:, 1, mid, mid, :], axis=(1))
        _d_avg_z = np.mean(dataset[-850:, 2, mid, mid, :], axis=(1))

        axs[0].plot(_t_axis, _d_avg_x, linewidth=lw,
                    label=dataset_legends[idx])

        axs[0].fill_between(_t_axis, _d_avg_x-_d_std_x, _d_avg_x
                            + _d_std_x, alpha=alpha, label=f'{dataset_legends[idx]} std. dev.')

        axs[1].plot(_t_axis, _d_avg_y, linewidth=lw,
                    label=dataset_legends[idx])
        axs[1].fill_between(_t_axis, _d_avg_y - _d_std_y, _d_avg_y
                            + _d_std_y, alpha=alpha, label=f'{dataset_legends[idx]} std. dev.')

        axs[2].plot(_t_axis, _d_avg_z, linewidth=lw,
                    label=dataset_legends[idx])
        axs[2].fill_between(_t_axis, _d_avg_z - _d_std_z, _d_avg_z
                            + _d_std_z, alpha=alpha, label=f'{dataset_legends[idx]} std. dev.')

    plt.xticks([0, 200, 400, 600, 800], fontproperties=FONT)
    fig.set_size_inches(7.85, 7)

    # get handles and labels
    handles, labels = axs[2].get_legend_handles_labels()
    print(labels)

    # specify order of items in legend
    order = [0, 3, 1, 4, 2, 5]
    order = [0, 1, 2, 3, 4, 5]

    # add legend to plot
    axs[2].legend([handles[idx] for idx in order], [labels[idx]
                                                    for idx in order], ncol=_n_datasets, prop=FONT, loc='lower left', bbox_to_anchor=(0, -0.55), fancybox=True, shadow=False)

    if unique_id is not None:
        if unique_id == 0:
            plt.show()
            return
        else:
            fig.savefig(
                f'plots/Plot_std_flow_profile_{save2file}{unique_id}.svg')
    else:
        fig.savefig(f'plots/Plot_std_flow_profile_{save2file}.svg')
    # plt.show()
    plt.close()


def plotPredVsTargKVS(input_pred, input_targ=None, input_lbm=None, file_name=None):
    """The plotPredVsTargKVS function aims to graphically compare model
    performance via plotting domain-wise averaged predicted and target
    velocities vs time. The standard deviations are additionally included for
    better comparison.

    Args:
        input_pred:
          Object of type PyTorch-Tensor containing the predicted dataset
        input_targ:
          Object of type PyTorch-Tensor containing the target dataset
        input_lbm:
          Object of type np-array containing the lbm dataset
        file_name:
          Object of type string containing unique file_name

    Returns:
        NONE:
          This function saves the graphical comparison to file.
    """
    t, c, x, y, z = input_pred.shape

    cell = 6
    t_max = t
    t_axis = np.arange(1, t_max+1)

    p_loc_x = input_pred[:, 0, cell, cell, cell]
    p_loc_y = input_pred[:, 1, cell, cell, cell]
    p_loc_z = input_pred[:, 2, cell, cell, cell]

    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)

    fig.suptitle(
        f'Velocity Components vs Time: {file_name}', fontsize=10)

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("$u_x$")
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].set_xlabel("t")
    axs[1].set_ylabel("$u_y$")
    axs[1].grid(axis='y', alpha=0.3)

    axs[2].set_xlabel("t")
    axs[2].set_ylabel("$u_z$")
    axs[2].grid(axis='y', alpha=0.3)

    axs[0].plot(t_axis, p_loc_x, linewidth=0.3, label='prediction')
    axs[1].plot(t_axis, p_loc_y, linewidth=0.3, label='prediction')
    axs[2].plot(t_axis, p_loc_z, linewidth=0.3, label='prediction')

    if input_targ is not None:
        t_loc_x = input_targ[:, 0, cell, cell, cell]
        t_loc_y = input_targ[:, 1, cell, cell, cell]
        t_loc_z = input_targ[:, 2, cell, cell, cell]

        axs[0].plot(t_axis, t_loc_x, linewidth=0.3, label='target')
        axs[1].plot(t_axis, t_loc_y, linewidth=0.3, label='target')
        axs[2].plot(t_axis, t_loc_z, linewidth=0.3, label='target')

    if input_lbm is not None:
        lbm_loc_x = input_lbm[100:, 1]
        lbm_loc_y = input_lbm[100:, 2]

        axs[0].plot(t_axis, lbm_loc_x, linewidth=0.3, label='lbm')
        axs[1].plot(t_axis, lbm_loc_y, linewidth=0.3, label='lbm')

    axs[0].legend(ncol=1, fontsize=9)
    axs[1].legend(ncol=1, fontsize=9)
    axs[2].legend(ncol=1, fontsize=9)

    fig.savefig(f'plots/Compare_loc_flow_profiles_{file_name}.png')
    plt.close()


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
    _x_rand_1 = np.random.rand(850, 3, 13, 13, 13) * 0.3
    _x_desc_1 = 'MD'
    _x_rand_2 = np.random.rand(850, 3, 13, 13, 13) * 0.3
    _x_desc_2 = 'MD + Hybrid ML'
    _x_rand_3 = np.random.rand(850, 3, 13, 13, 13) * 0.3
    _x_desc_3 = 'Hybrid ML only'
    _x_file = 'file_name'

    plot_flow_profile_std(np_datasets=[_x_rand_1, _x_rand_2, _x_rand_3], dataset_legends=[
                      _x_desc_1, _x_desc_2, _x_desc_3], save2file=_x_file, unique_id=0)
