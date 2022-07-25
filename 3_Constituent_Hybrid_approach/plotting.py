import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from utils_new import csv2dataset, csv2dataset_mp, mamico_csv2dataset, mamico_csv2dataset_mp
mpl.use('Agg')
plt.style.use(['science'])
np.set_printoptions(precision=2)


def getColor(c, N, idx):
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


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
                          alpha=0.8, marker='.', s=0.25, vmin=-2, vmax=2, cmap=cm)

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
    fig.set_size_inches(8, 8)
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


def visualizeMaMiCoDataset(file_names, dataset_names, u_wall):
    # BRIEF: This function is used to visualize the MaMiCo generated simulation
    # data. It loads the dataset from a csv file.
    # Here, the use case is tailored to simulation results containing 1000
    # timesteps in a 26 x 26 x 26 spatial domain. Hence the default values.
    # PARAMETERS:
    # filename -  the name of the file of interest including file suffix,
    # e.g. 'my_values.csv'

    for i in range(len(file_names)):
        # Load dataset from csv
        print('Loading Dataset.')
        _dataset = mamico_csv2dataset(file_names[i])
        print('Complete.')
        # Create 3D scatterplot for meaningful timesteps.
        print('Creating ColorMap.')
        colorMap(_dataset, dataset_names[i])
        print('Complete.')

        print('Creating flowProfile.')
        flowProfile(_dataset, dataset_names[i], u_wall[i])
        print('Complete.')
    return


def plotAvgLoss(avg_losses, file_prefix=0, file_name=0):
    # BRIEF: This function is used to visualize the losses, in other words
    # chart model learning.
    # PARAMETERS:
    # avg_losses - list containing the average loss from each loading cycle
    # file_name - designated name to save file

    x_axis = range(1, (len(avg_losses)+1), 1)

    x_ticks = [1]

    if len(avg_losses) % 10 == 0:
        for i in range(int(len(avg_losses)/10) + 1):
            x_ticks.append(10*i)
    else:
        for i in range(int(len(avg_losses)/10) + 2):
            x_ticks.append(10*i)

    y_ticks = np.arange(0, 0.176, 0.025)
    # max_x = len(min_losses)
    # max_loss = max(max_losses)

    fig, (ax1) = plt.subplots(1, constrained_layout=True)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Error')
    ax1.plot(x_axis, avg_losses, label='Average Loss')
    ax1.set_yticks(y_ticks)
    ax1.set_xticks(x_ticks)
    fig.set_size_inches(6, 3.5)
    # plt.show()
    if file_name != 0:
        fig.savefig(f'{file_prefix}Plot_Avg_Losses_{file_name}.svg')

    plt.close()


def compareAvgLoss(loss_files, loss_labels, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    losses = csv2dataset_mp(loss_files)

    loss_list = []
    for loss in losses:
        loss_list.append(loss)

    # num_epoch = (loss_list[0]).shape[0]

    # x_axis = range(1, (num_epoch+1), 1)

    fig, (ax1) = plt.subplots(1, constrained_layout=True)
    for idx, loss in enumerate(loss_list):
        ax1.plot(range(1, ((loss_list[idx]).shape[0]+1), 1), loss, color=getColor(c='tab20',
                 N=12, idx=idx), label=loss_labels[idx])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.legend(ncol=2, fontsize=9, loc='lower left')
    fig.set_size_inches(6, 2)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Avg_Losses_{file_name}.svg')

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
    ax1.legend(ncol=3, fontsize=9)
    fig.set_size_inches(7, 2.5)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Loss_vs_Valid_{file_name}.svg')

    plt.close()


def compareAvgLossRNN(l_of_l_files, l_of_l_labels, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    list_of_list_l = []
    list_of_LR = ['0.001', '0.0005', '0.0001',
                  '0.00005', '0.00001', '0.000005']

    for list in l_of_l_files:
        losses = csv2dataset_mp(list)
        list_l = []

        for loss in losses:
            list_l.append(loss)

        list_of_list_l.append(list_l)

    num_epoch = list_of_list_l[0][0].shape[0]+1

    x_axis = range(1, num_epoch, 1)
    print(len(list_of_list_l))
    fig, axs = plt.subplots(len(list_of_list_l),
                            sharex=True, constrained_layout=True)
    # axs = [axs]
    for i, list_of_loss in enumerate(list_of_list_l):
        for j, loss in enumerate(list_of_loss):
            axs[i].plot(x_axis, loss, color=getColor(
                c='tab20', N=12, idx=j), label=l_of_l_labels[i][j])

        axs[i].set_title(f'Learning Rate = {list_of_LR[i]}')
        axs[i].set_ylabel('Avg Loss')
        # axs[i].set_xlabel('Number of Epochs')
        # axs[i].legend(ncol=4, fontsize=9)
        axs[i].grid(axis='y', alpha=0.3)

    axs[-1].set_xlabel('Epoch')
    axs[-1].legend(ncol=3, fontsize=9)
    fig.set_size_inches(6, 10)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Avg_Losses_{file_name}.svg')

    plt.close()


def compareLossVsValidRNN(l_of_l_files, l_of_l_labels, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    list_of_list_l = []
    list_of_layers = [1, 2, 3, 4]
    for list in l_of_l_files:
        losses = csv2dataset_mp(list)
        list_l = []

        for loss in losses:
            list_l.append(loss)

        list_of_list_l.append(list_l)

    num_epoch = list_of_list_l[0][0].shape[0]

    x_axis = range(1, (num_epoch+1), 1)

    fig, axs = plt.subplots(len(list_of_list_l), sharex=True,
                            sharey=True, constrained_layout=True)

    for i, list_of_loss in enumerate(list_of_list_l):
        for j in range(int(len(list_of_loss)/2)):
            axs[i].plot(x_axis, list_of_loss[2*j], color=getColor(
                c='tab20', N=12, idx=j), label=l_of_l_labels[i][2*j])
            axs[i].plot(x_axis, list_of_loss[2*j+1], color=getColor(
                c='tab20', N=12, idx=j), linestyle='--', label=l_of_l_labels[i][2*j+1])

        axs[i].set_title(f'Number of Layers = {list_of_layers[i]}')
        axs[i].set_ylabel('Avg Loss')
        axs[i].grid(axis='y', alpha=0.3)

    axs[-1].set_xlabel('Epoch')
    axs[-1].legend(ncol=3, fontsize=9)
    fig.set_size_inches(6, 6.66)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Avg_Losses_{file_name}.svg')

    plt.close()


def compareColorMap(preds, targs, model_name, dataset_name):
    t, c, d, h, w = preds.shape
    steps = np.arange(0, h).tolist()
    X, Y, Z = np.meshgrid(steps, steps, steps)
    t_samples = [60, 125, 250, 500, 999]

    # Creating color map
    cm = plt.cm.get_cmap('Spectral')

    # Creating figure
    fig = plt.figure()
    fig.suptitle('Visualization of Target vs Prediction', fontsize=16)
    directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/'

    counter = 1

    # Creating subplots
    for counter in range(1, 6):
        ax_pred = fig.add_subplot(5, 2, (2*counter-1), projection='3d')
        ax_pred.set_title(
            f'Prediction for t={t_samples[counter-1]}', fontsize=10)
        sc = ax_pred.scatter3D(X, Y, Z, c=preds[[counter-1], 0, :, :, :],
                               alpha=0.8, marker='.', s=0.25, vmin=-2, vmax=6, cmap=cm)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_zticks([])
        ax_pred.grid(False)

        ax_targ = fig.add_subplot(5, 2, (2*counter), projection='3d')
        ax_targ.set_title(f'Target for t={t_samples[counter-1]}', fontsize=10)
        sc = ax_targ.scatter3D(X, Y, Z, c=targs[[counter-1], 0, :, :, :],
                               alpha=0.8, marker='.', s=0.25, vmin=-2, vmax=6, cmap=cm)
        ax_targ.set_xticks([])
        ax_targ.set_yticks([])
        ax_targ.set_zticks([])
        ax_targ.grid(False)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    fig.set_size_inches(6, 10.5)
    fig.savefig(
        f'{directory}Colormap_Comparison_{model_name}_{dataset_name}.png')
    # fig.savefig('myfig.eps', format='eps')
    # plt.show()
    plt.close()

    pass


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
    plt.legend(ncol=2, fontsize=9)
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


def compareFlowProfile3x3(preds, targs, model_id='', dataset_id=''):
    # BRIEF:
    #
    # PARAMETERS:
    # preds -
    # targs -
    # model_descriptor -
    # directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/1_UNET_AE/'
    # directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/5_Hybrid_Couette/'
    directory = '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/7_Hybrid_Both/'

    t, c, d, h, w = preds.shape
    steps = np.arange(0, d).tolist()
    pred_avg_050 = [[], [], []]
    pred_avg_500 = [[], [], []]
    pred_avg_999 = [[], [], []]
    targ_avg_050 = [[], [], []]
    targ_avg_500 = [[], [], []]
    targ_avg_999 = [[], [], []]

    for i in range(3):
        for j in range(w):
            pred_avg_050[i].append(preds[50, i, :, :, j].mean())
            targ_avg_050[i].append(targs[50, i, :, :, j].mean())
            pred_avg_500[i].append(preds[500, i, :, :, j].mean())
            targ_avg_500[i].append(targs[500, i, :, :, j].mean())
            pred_avg_999[i].append(preds[-1, i, :, :, j].mean())
            targ_avg_999[i].append(targs[-1, i, :, :, j].mean())

    max_ux = int(max(targ_avg_999[0])) + 1

    preds_avg = [pred_avg_050, pred_avg_500, pred_avg_999]
    targs_avg = [targ_avg_050, targ_avg_500, targ_avg_999]
    samples = [50, 500, 999]
    # time_list = [0, 4, 1, 5, 2, 6, 3, 7]

    fig, axs = plt.subplots(3, 3, sharex=True)  # , sharey=True)
    # fig.suptitle('Target and Prediction Comparison', fontsize=16)
    # plt.tick_params(labelcolor='none', which='both', top=False,
    #                 bottom=False, left=False, right=False)
    plt.setp(axs[-1, :], xlabel='Z-Direction')  # , fontsize=12)
    plt.setp(axs[0, 0], ylabel='$u_x$')  # , fontsize=12)
    plt.setp(axs[1, 0], ylabel='$u_y$')  # , fontsize=12)
    plt.setp(axs[2, 0], ylabel='$u_z$')  # , fontsize=12)
    fig.set_size_inches(10, 8)
    # plt.yticks(range(-2, 7, 2))
    # plt.xticks([])

    for i, row in enumerate(axs):
        axs[i, 0].plot(steps, preds_avg[0][i], label='Avg Prediction')
        axs[i, 0].plot(steps, targs_avg[0][i], label='Avg Target')
        axs[i, 0].set_yticks(list(np.arange(-0.25, 0.255, 0.25)))

        axs[i, 1].plot(steps, preds_avg[1][i], label='Avg Prediction')
        axs[i, 1].plot(steps, targs_avg[1][i], label='Avg Target')
        axs[i, 1].set_yticks(list(np.arange(-0.25, 0.255, 0.25)))

        axs[i, 2].plot(steps, preds_avg[2][i], label='Avg Prediction')
        axs[i, 2].plot(steps, targs_avg[2][i], label='Avg Target')
        axs[i, 2].set_yticks(list(np.arange(-0.25, 0.255, 0.25)))

        axs[0, i].set_title(f'Timestep {samples[i]}', fontsize=12)
        '''
        axs[i, 0].plot(steps, preds_avg[i][0], label='Avg Prediction')
        axs[i, 0].plot(steps, targs_avg[i][0], label='Avg Target')
        axs[i, 0].set_yticks(list(np.arange(-0.5, 1.55, 0.5)))
        axs[i, 1].set_title(f'Timestep {samples[i]}')
        axs[i, 1].plot(steps, preds_avg[i][1], label='Avg Prediction')
        axs[i, 1].plot(steps, targs_avg[i][1], label='Avg Target')
        axs[i, 1].set_yticks(list(np.arange(-0.25, 0.255, 0.25)))
        '''

    axs[0, 0].set_yticks(list(np.arange(-1, max_ux, 1)))
    axs[0, 1].set_yticks(list(np.arange(-1, max_ux, 1)))
    axs[0, 2].set_yticks(list(np.arange(-1, max_ux, 1)))
    axs[-1, -1].legend(ncol=1, fontsize=12)
    fig.savefig(
        f'{directory}CompareFlowprofile_{model_id}_{dataset_id}.svg')
    plt.close()
    pass


def compareErrorTimeline_csv(l_of_l_files, l_of_l_labels, l_of_titles, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    list_of_list_l = []

    for list in l_of_l_files:
        losses = csv2dataset_mp(list)
        list_l = []

        for loss in losses:
            list_l.append(loss)

        list_of_list_l.append(list_l)

    num_epoch = list_of_list_l[0][0].shape[0]+1

    x_axis = range(1, num_epoch, 1)

    fig, axs = plt.subplots(len(list_of_list_l),
                            sharex=True, constrained_layout=True)

    for i, list_of_loss in enumerate(list_of_list_l):
        for j, loss in enumerate(list_of_loss):
            axs[i].plot(x_axis, loss, color=getColor(
                c='tab20', N=12, idx=j*2), label=l_of_l_labels[j])

            axs[i].set_title(f'{l_of_titles[i]}')
            axs[i].set_ylabel('Error')
            # axs[i].set_xlabel('Number of Epochs')
            # axs[i].legend(ncol=4, fontsize=9)
            axs[i].grid(axis='y', alpha=0.3)

    axs[-1].set_xlabel('Timestep')
    axs[-1].legend(ncol=3, fontsize=9)
    fig.set_size_inches(6, 7)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Error_Timeline_{file_name}.svg')

    plt.close()


def compareErrorTimeline_np(l_of_l_losses, l_of_l_labels, l_of_titles, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:

    for i, list_of_l in enumerate(l_of_l_losses):
        for j, label in enumerate(l_of_l_labels):
            print(
                f'Length of {label} list for {l_of_titles[i]} dataset: {len(l_of_l_losses[i][j])}')

    num_epoch = len(l_of_l_losses[0][0])

    x_axis = range(1, num_epoch+1, 1)

    fig, axs = plt.subplots(len(l_of_l_losses),
                            sharex=True, constrained_layout=True)

    for i, list_of_loss in enumerate(l_of_l_losses):
        for j, loss in enumerate(list_of_loss):
            axs[i].plot(x_axis, loss, color=getColor(
                c='tab20', N=12, idx=j*4), linewidth=0.2, label=l_of_l_labels[j])

            axs[i].set_title(f'{l_of_titles[i]}')
            axs[i].set_ylabel('Error')
            # axs[i].set_xlabel('Number of Epochs')
            # axs[i].legend(ncol=4, fontsize=9)
            axs[i].grid(axis='y', alpha=0.3)

    axs[-1].set_xlabel('Timestep')
    axs[-1].legend(ncol=3, fontsize=9)
    fig.set_size_inches(6, 10)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Error_Timeline_{file_name}.svg')

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


def plotVelocityField(input_1, input_2='void', file_prefix=0, file_name=0):

    t, c, x, y, z = input_1.shape
    print(input_1.shape)
    X, Z = np.meshgrid(np.arange(0, x, 1), np.arange(0, z, 1))
    t_samples = [0, int(t/2), t-1]
    columns = 2
    d = int(y/2)
    # inputs = [input_1.mean(axis=3), input_2.mean(axis=3)]
    inputs = [input_1, input_2]
    if input_2 == 'void':
        columns = 1
    # set color field for better visualisation
    # n = -2
    # color = np.sqrt(((v-n)/2)*2 + ((u-n)/2)*2)

    # set plot parameters
    # u - velocity component in x-direction
    # v - velocity component in y-direction
    fig, axs = plt.subplots(3, columns, constrained_layout=True)

    if input_2 == 'void':
        print(axs.shape)
        axs = np.reshape(axs, (3, 1))
        print(axs.shape)

    for i in range(3):
        for j in range(columns):
            u_x = inputs[j][t_samples[i], 0, :, d, :]
            print(u_x.shape)
            u_z = inputs[j][t_samples[i], 2, :, d, :]
            print(u_x.shape)
            axs[i][j].quiver(X, Z, u_x, u_z, units='width')
            # axs[i][j].xaxis.set_major_locator(plt.NullLocator())
            # axs[i][j].yaxis.set_major_locator(plt.NullLocator())
            axs[i][0].set_ylabel('Height $z$')
            axs[-1][j].set_xlabel('Depth $x$')

    fig.suptitle('KVS Velocity Field (Cross-Section)')
    axs[0][0].set_title('Prediction')
    axs[0][-1].set_title('Target')

    fig.set_size_inches(6, 10)
    if file_name != 0:
        fig.savefig(
            f'{file_prefix}Plot_Velocity_Field_{file_name}.svg')
    plt.show()


def plotPredVsTargKVS(input_1, input_2='void', file_prefix=0, file_name=0):
    t, c, x, y, z = input_1.shape
    mid = int(x/2)
    t_max = 100
    t_axis = np.arange(1, t_max+1)

    p_std = np.std(input_1[:t_max, 2, mid, :, mid], axis=1)
    t_std = np.std(input_2[:t_max, 2, mid, :, mid], axis=1)

    p_avg = np.mean(input_1[:t_max, 2, mid, :, mid], axis=1)
    t_avg = np.mean(input_2[:t_max, 2, mid, :, mid], axis=1)

    p_loc = input_1[:t_max, 2, mid, mid, mid]
    t_loc = input_2[:t_max, 2, mid, mid, mid]

    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    axs[0].plot(t_axis, p_avg, linewidth=0.5, label='Prediction')
    axs[0].fill_between(t_axis, p_avg-p_std, p_avg+p_std,
                        alpha=0.2, label='Prediction')
    axs[0].plot(t_axis, t_avg, linewidth=0.5, label='Target')
    axs[0].fill_between(t_axis, t_avg-t_std, t_avg+t_std,
                        alpha=0.2, label='Prediction')
    axs[0].set_ylabel('Averaged $u_z$')
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].plot(t_axis, p_loc, linewidth=0.5, label='Prediction')
    axs[1].plot(t_axis, t_loc, linewidth=0.5, label='Target')
    axs[1].set_ylabel(f'Local $u_z$ at [t, {mid}, {mid}, {mid}]')
    axs[1].grid(axis='y', alpha=0.3)
    axs[1].set_xlabel('Timestep')
    axs[1].legend(ncol=2, fontsize=9)

    fig.set_size_inches(6, 6)
    if file_name != 0:
        fig.savefig(
            f'{file_prefix}Plot_PredVsTarg_KVS_{file_name}.svg')
    # plt.show()
    plt.close()
    pass


def plotPredVsTargCouette(input_1, input_2='void', file_prefix=0, file_name=0):
    t, c, x, y, z = input_1.shape
    mid = int(x/2)
    t_max = t
    t_axis = np.arange(1, t_max+1)

    p_std = np.std(input_1[:t_max, 0, mid, :, mid], axis=1)
    t_std = np.std(input_2[:t_max, 0, mid, :, mid], axis=1)

    p_avg = np.mean(input_1[:t_max, 0, mid, :, mid], axis=1)
    t_avg = np.mean(input_2[:t_max, 0, mid, :, mid], axis=1)

    p_loc = input_1[:t_max, 0, mid, mid, mid]
    t_loc = input_2[:t_max, 0, mid, mid, mid]

    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    axs[0].plot(t_axis, p_avg, linewidth=0.5, label='Prediction')
    axs[0].fill_between(t_axis, p_avg-p_std, p_avg+p_std,
                        alpha=0.2, label='Prediction')
    axs[0].plot(t_axis, t_avg, linewidth=0.5, label='Target')
    axs[0].fill_between(t_axis, t_avg-t_std, t_avg+t_std,
                        alpha=0.2, label='Prediction')
    axs[0].set_ylabel('Averaged $u_x$')
    axs[0].grid(axis='y', alpha=0.3)

    axs[1].plot(t_axis, p_loc, linewidth=0.5, label='Prediction')
    axs[1].plot(t_axis, t_loc, linewidth=0.5, label='Target')
    axs[1].set_ylabel(f'Local $u_x$ at [t, {mid}, {mid}, {mid}]')
    axs[1].grid(axis='y', alpha=0.3)
    axs[1].set_xlabel('Timestep')
    axs[1].legend(ncol=2, fontsize=9)

    fig.set_size_inches(6, 6)
    if file_name != 0:
        fig.savefig(
            f'{file_prefix}Plot_PredVsTarg_Couette_{file_name}.svg')
    # plt.show()
    plt.close()
    pass


def main():
    pass


if __name__ == "__main__":
    _dir = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/CleanCouette/Testing/'
    _file_names = [
        f'{_dir}clean_couette_test_combined_domain_6_0_bottom.csv',
        f'{_dir}clean_couette_test_combined_domain_6_0_middle.csv',
        f'{_dir}clean_couette_test_combined_domain_6_0_top.csv'
    ]
    _dataset_names = [
        'C-6-0-B',
        'C-6-0-M',
        'C-6-0-T'
    ]
    _u_wall = [6, 6, 6]
    visualizeMaMiCoDataset(
        file_names=_file_names,
        dataset_names=_dataset_names,
        u_wall=_u_wall
    )
