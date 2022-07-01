from utils import mamico_csv2dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from utils import csv2dataset, csv2dataset_mp
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

    num_epoch = (loss_list[0]).shape[0]

    x_axis = range(1, (num_epoch+1), 1)

    fig, (ax1) = plt.subplots(1, constrained_layout=True)
    for idx, loss in enumerate(loss_list):
        ax1.plot(x_axis, loss, color=getColor(c='tab20',
                 N=12, idx=idx), label=loss_labels[idx])

    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Error')
    ax1.legend(ncol=2, fontsize=7)
    fig.set_size_inches(7, 2.5)
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
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Error')
    ax1.legend(ncol=2, fontsize=7)
    fig.set_size_inches(7, 2.5)
    if file_name != 0:
        fig.savefig(f'{file_prefix}Compare_Loss_vs_Valid_{file_name}.svg')

    plt.close()


def compareAvgLossRNN(l_of_l_files, l_of_l_labels, file_prefix=0, file_name=0):
    # BRIEF:
    # PARAMETERS:
    list_of_list_l = []

    for list in l_of_l_files:
        losses = csv2dataset_mp(list)
        list_l = []

        for loss in losses:
            list_l.append(loss)

        list_of_list_l.append(list_l)

    num_epoch = 30

    x_axis = range(1, (num_epoch+1), 1)

    fig, axs = plt.subplots(len(list_of_list_l), constrained_layout=True)

    for i, list_of_loss in enumerate(list_of_list_l):
        for j in range(int(len(list_of_loss)/2)):
            axs[i].plot(x_axis, list_of_loss[2*j], color=getColor(
                c='tab20', N=12, idx=j), label=l_of_l_labels[i][2*j])
            axs[i].plot(x_axis, list_of_loss[2*j+1], color=getColor(
                c='tab20', N=12, idx=j), linestyle='--', label=l_of_l_labels[i][2*j+1])

        axs[i].set_xlabel('Number of Epochs')
        axs[i].set_ylabel('Error')
        axs[i].legend(ncol=1, fontsize=7)

    fig.set_size_inches(7, 10)
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


def main():

    pass


if __name__ == "__main__":

    visualizeMaMiCoDataset(
        filenames=['clean_kvs_test_combined_domain.csv'], dataset_names=['kvs test'], u_wall=[0])
