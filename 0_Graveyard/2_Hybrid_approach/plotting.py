from utils import mamico_csv2dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
mpl.use('Agg')
plt.style.use(['science'])
np.set_printoptions(precision=2)


def colorMap(dataset, dataset_name):
    # BRIEF: This function is used to visualize a dataset in a 3D
    # scatterplot, aka color map. Here, the use case is tailored
    # to simulation results containing 1000 timesteps in a 26x26x26
    # spatial domain. To this end, 8 timesteps will be considered:
    # [0, 25, 50, 100, 200, 400, 800, 999]
    #
    # PARAMETERS:
    # dataset - contains the MaMiCoDataset [1000, 3, 26, 26, 26]
    # dataset_name - refers to a unique numeric identifier

    t, c, d, h, w = dataset.shape
    steps = np.arange(0, h).tolist()
    X, Y, Z = np.meshgrid(steps, steps, steps)
    counter = 0
    t = [0, 25, 50, 100, 200, 400, 600, 800, 999]

    # Creating color map
    cm = plt.cm.get_cmap('Spectral')

    # Creating figure
    fig = plt.figure()
    fig.suptitle(f'Visualization of Dataset: {dataset_name}', fontsize=16)

    # Creating subplots
    while counter < 9:
        ax = fig.add_subplot(3, 3, (counter+1), projection='3d')
        ax.set_title(f't={t[counter]}', fontsize=10)
        sc = ax.scatter3D(X, Y, Z, c=dataset[t[counter], 0, :, :, :],
                          alpha=0.8, marker='.', s=0.25, vmin=-4, vmax=4, cmap=cm)

        if counter == 4:
            ax.set_xlabel("X", fontsize=7, fontweight='bold')
            ax.set_ylabel("Y", fontsize=7, fontweight='bold')
            ax.set_zlabel("Z", fontsize=7, fontweight='bold')
            ax.xaxis.labelpad = -10
            ax.yaxis.labelpad = -10
            ax.zaxis.labelpad = -10

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        counter += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    # fig.set_size_inches(3.5, 2)
    fig.savefig(f'Colormap_Visualization_Dataset_{dataset_name}.png')
    # fig.savefig('myfig.eps', format='eps')
    # plt.show()
    plt.close()


def flowProfile(dataset, dataset_name, u_wall=1):
    # BRIEF: This function is used to visualize a dataset in a 2D
    # flow profile. Here, the use case is tailored to simulation
    # results containing 1000 timesteps in a 26x26x26 spatial domain.
    # To this end, 9 timesteps will be considered for each velocity
    # component [0, 25, 50, 100, 200, 400, 600, 800, 999]
    #
    # PARAMETERS:
    # dataset - contains the MaMiCoDataset [1000, 3, 26, 26, 26]
    # dataset_name - refers to a unique numeric identifier
    # u_wall - the speed of the moving wall for the specific couette
    # scenario

    t, c, d, h, w = dataset.shape
    steps = np.arange(0, h).tolist()

    mid = int(h/2)
    samples = [0, 25, 50, 100, 200, 400, 600, 800, 999]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, sharey=True, constrained_layout=True)  # sharex=True
    # plt.ylabel('Velocity $u$')
    fig.suptitle('Flow Profiles in X-,Y- and Z-Direction', fontsize=10)

    # Plot all u_x entries along the central x-axis
    ax1.set_xlabel("X")
    ax1.set_ylabel("$u_x$")
    # Plot all u_x entries along the central Y-axis
    ax2.set_xlabel("Y")
    ax2.set_ylabel("$u_x$")
    # Plot all u_x entries along the central Z-axis
    ax3.set_xlabel("Z")
    ax3.set_ylabel("$u_x$")

    for time in samples:
        ax1.plot(steps, dataset[time, 0, :, mid, mid], label=f't = {time}')
        ax2.plot(steps, dataset[time, 0, mid, :, mid], label=f't = {time}')
        ax3.plot(steps, dataset[time, 0, mid, mid, :], label=f't = {time}')

    plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
    # plt.xlabel('Spatial Dimension')
    plt.legend(ncol=4, fontsize=7)

    fig.savefig(f'Flowprofile_Visualization_Dataset_{dataset_name}.png')
    plt.close()


def visualizeMaMiCoDataset(filenames, dataset_names, u_wall):
    # BRIEF: This function is used to visualize the MaMiCo generated simulation
    # data. It loads the dataset from a csv file.
    # Here, the use case is tailored to simulation results containing 1000
    # timesteps in a 26 x 26 x 26 spatial domain. Hence the default values.
    # PARAMETERS:
    # filename -  the name of the file of interest including file suffix,
    # e.g. 'my_values.csv'

    for i in range(len(filenames)):
        # Load dataset from csv
        print('Loading Dataset.')
        _dataset = mamico_csv2dataset(filenames[i])
        print('Complete.')
        # Create 3D scatterplot for meaningful timesteps.
        print('Creating ColorMap.')
        colorMap(_dataset, dataset_names[i])
        print('Complete.')

        print('Creating flowProfile.')
        flowProfile(_dataset, dataset_names[i], u_wall[i])
        print('Complete.')
    return


def plotMinMaxAvgLoss(min_losses, avg_losses, max_losses, file_name=0):
    # BRIEF: This function is used to visualize the losses, in other words
    # chart model learning. Note that this does not consider epochs, but
    # but rather individual loading cycles. e.g. 1 epoch = n * loading cycles
    # where n is the number of loaders.
    # PARAMETERS:
    # min_losses - list containing the minimum loss from each loading cycle
    # avg_losses - list containing the maximum loss from each loading cycle
    # max_losses - list containing the average loss from each loading cycle
    # file_name - designated name to save file

    x_axis = range(1, (len(min_losses)+1), 1)

    x_ticks = [1]

    if len(min_losses) % 10 == 0:
        for i in range(int(len(min_losses)/10) + 1):
            x_ticks.append(10*i)
    else:
        for i in range(int(len(min_losses)/10) + 2):
            x_ticks.append(10*i)

    y_ticks = []
    if max(max_losses) > 5:
        y_ticks = np.arange(0, 6.01, 1)
    elif max(max_losses) > 4:
        y_ticks = np.arange(0, 5.01, 1)
    elif max(max_losses) > 3:
        y_ticks = np.arange(0, 4.01, 0.5)
    elif max(max_losses) > 2:
        y_ticks = np.arange(0, 3.01, 0.5)
    elif max(max_losses) > 1:
        y_ticks = np.arange(0, 2.01, 0.25)
    elif max(max_losses) > 0.5:
        y_ticks = np.arange(0, 1.01, 0.1)
    elif max(max_losses) > 0.3:
        y_ticks = np.arange(0, 0.51, 0.05)
    # max_x = len(min_losses)
    # max_loss = max(max_losses)

    fig, (ax1) = plt.subplots(1, constrained_layout=True)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Error')
    ax1.plot(x_axis, min_losses, ':', label='Minimum Loss')
    ax1.plot(x_axis, avg_losses, label='Average Loss')
    ax1.plot(x_axis, max_losses, ':', label='Maximum Loss')
    ax1.set_yticks(y_ticks)
    ax1.set_xticks(x_ticks)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    fig.set_size_inches(6, 3.5)
    # plt.show()
    if file_name != 0:
        fig.savefig(f'MinMaxAvgLosses_{file_name}.svg')

    plt.close()


def compareFlowProfile(preds, targs, model_descriptor):
    # BRIEF:
    #
    # PARAMETERS:
    # preds -
    # targs -
    # model_descriptor -

    t, c, d, h, w = preds.shape
    steps = np.arange(0, h).tolist()
    samples = [0, 25, 50, 100, 200, 400, 800, 999]
    time_list = [0, 4, 1, 5, 2, 6, 3, 7]
    mid = int(h/2)

    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
    fig.suptitle('Target and Prediction Comparison', fontsize=16)
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)
    plt.setp(axs[-1, :], xlabel='Z-Direction')
    plt.setp(axs[:, 0], ylabel='$u_x$')

    axs = axs.ravel()
    plt.yticks(range(-2, 7, 2))
    plt.xticks([])

    for i in range(8):
        axs[i].plot(steps, preds[time_list[i], 0, :,
                    mid, mid], label=f'Prediction')
        axs[i].plot(steps, targs[time_list[i], 0,
                    mid, :, mid], label=f'Target')
        axs[i].set_title(f't={samples[time_list[i]]}', fontsize=12)

    # plt.xlabel('Spatial Dimension')
    plt.legend(ncol=2, fontsize=7)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    fig.set_size_inches(12, 9)

    fig.savefig(
        f'CompareFlowprofile_Validation_Dataset_{model_descriptor}.svg')
    plt.close()


def showSample():
    v_step = 20 / (31)
    v_steps = np.arange(0, 20 + v_step, v_step).tolist()
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    line = np.linspace(-5, 25, 100)
    middle = 10 * np.ones(100, )
    ax.plot3D(middle, middle, line, 'black')
    ax.plot3D(middle, line, middle, 'black')
    ax.plot3D(line, middle, middle, 'black')
    ax.set_xlabel("X", fontsize=7, fontweight='bold')
    ax.set_ylabel("Z", fontsize=7, fontweight='bold')
    ax.set_zlabel("Y", fontsize=7, fontweight='bold')
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    plt.show()
    fig.savefig('Plots/Sampling_of_Volume.png')


def rubbish():
    '''
    def colorMap2():
        t = 1000
        u = 10
        w = 20
        n = 2
        v = 31
        sigma = 0.3
        seed = 1
        analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                       nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
        t, c, d, h2, w2 = analytical.shape
        v_step = w / (h2-1)
        v_steps = np.arange(0, w + v_step, v_step).tolist()
        X, Y, Z = np.meshgrid(v_steps, v_steps, v_steps)
        U = [analytical[1, 0, :, :, :], analytical[50, 0, :, :, :],
             analytical[250, 0, :, :, :], analytical[999, 0, :, :, :]]
        titles = ['T = 1', 'T = 50', 'T = 250', 'T = 999', ]
        cm = plt.cm.get_cmap('Spectral')
        # Creating figure
        fig = plt.figure()
        for i in range(len(U)):
            ax = fig.add_subplot(2, 2, (i+1), projection='3d')
            ax.set_title(titles[i], fontsize=10)
            sc = ax.scatter3D(Z, Y, X, c=U[i],
                              alpha=0.8, marker='.', s=0.25, vmin=-4, vmax=14, cmap=cm)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sc, cax=cbar_ax)

        plt.show()
        fig.savefig('Plots/Sample_Volume_Couette_Noisy.png')
    '''
    pass


def test():
    _filenames = [
        'clean_couette_test_combined_domain_0_5_top.csv',
        'clean_couette_test_combined_domain_0_5_middle.csv',
        'clean_couette_test_combined_domain_0_5_bottom.csv',
        'clean_couette_test_combined_domain_1_0_top.csv',
        'clean_couette_test_combined_domain_1_0_middle.csv',
        'clean_couette_test_combined_domain_1_0_bottom.csv',
        'clean_couette_test_combined_domain_2_0_top.csv',
        'clean_couette_test_combined_domain_2_0_middle.csv',
        'clean_couette_test_combined_domain_2_0_bottom.csv',
        'clean_couette_test_combined_domain_3_0_top.csv',
        'clean_couette_test_combined_domain_3_0_middle.csv',
        'clean_couette_test_combined_domain_3_0_bottom.csv',
        'clean_couette_test_combined_domain_4_0_top.csv',
        'clean_couette_test_combined_domain_4_0_middle.csv',
        'clean_couette_test_combined_domain_4_0_bottom.csv',
        'clean_couette_test_combined_domain_5_0_top.csv',
        'clean_couette_test_combined_domain_5_0_middle.csv',
        'clean_couette_test_combined_domain_5_0_bottom.csv'
    ]
    _dataset_names = [
        '0_5_top',
        '0_5_middle',
        '0_5_bottom',
        '1_0_top',
        '1_0_middle',
        '1_0_bottom',
        '2_0_top',
        '2_0_middle',
        '2_0_bottom',
        '3_0_top',
        '3_0_middle',
        '3_0_bottom',
        '4_0_top',
        '4_0_middle',
        '4_0_bottom',
        '5_0_top',
        '5_0_middle',
        '5_0_bottom'
    ]
    _u_wall = [
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        4.0,
        4.0,
        4.0,
        5.0,
        5.0,
        5.0
    ]
    visualizeMaMiCoDataset(
        filenames=_filenames,
        dataset_names=_dataset_names,
        u_wall=_u_wall
    )


def main():

    pass


if __name__ == "__main__":
    # test()
    _preds = torch.randn(8, 3, 18, 18, 18) + 1
    _targs = torch.randn(8, 3, 18, 18, 18) + 3

    compareFlowProfile(preds=_preds, targs=_targs, model_descriptor='1_1_1_1')
