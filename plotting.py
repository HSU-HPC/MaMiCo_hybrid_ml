import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from couette_solver import my3DCouetteSolver
plt.style.use(['science'])
np.set_printoptions(precision=2)


def save3DArray2File(input_array, prediction):
    # 1) Convert 3D array to 2D array
    input_reshaped = input_array.reshape(input_array.shape[0], -1)

    # 2) Save 2D array to file
    t, c, x, y = input_array.shape
    name = f'{prediction}_{t}_{x}_{y}'
    np.savetxt(f'{name}.csv', input_reshaped)


def load3DArrayFromFile(input_file, input_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(f'{input_file}')

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(
        loaded_array.shape[0], loaded_array.shape[1] // input_shape[2], input_shape[2])
    return original_array


def checkSaveLoad(input_array, loaded_array):
    print("shape of input array: ", input_array.shape)
    print("shape of loaded array: ", loaded_array.shape)

    if (input_array == loaded_array).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")


def plotFlowProfile(input_array, wall_height=20, u_wall=10):

    if input_array.ndim == 2:
        t, h = input_array.shape

        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1) = plt.subplots(1, sharey=True, constrained_layout=True)
        # fig.suptitle('Flow Profiles in X- and Y-Direction', fontsize=10)

        ax1.set_title('Flow Profile in Y-Direction', fontsize=10)
        ax1.plot(v_steps, input_array[0, :], label='t = 0')
        ax1.plot(v_steps, input_array[int(
            0.01*t), :], label=f't = {int(0.01*t)}')
        ax1.plot(v_steps, input_array[int(
            0.05*t), :], label=f't = {int(0.05*t)}')
        ax1.plot(v_steps, input_array[int(
            0.10*t), :], label=f't = {int(0.1*t)}')
        ax1.plot(v_steps, input_array[int(
            0.25*t), :], label=f't = {int(0.25*t)}')
        ax1.plot(v_steps, input_array[int(
            0.50*t), :], label=f't = {int(0.5*t)}')
        ax1.plot(v_steps, input_array[int(
            0.75*t), :], label=f't = {int(0.75*t)}')
        ax1.plot(v_steps, input_array[-1, :], label=f't = {t}')

        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.ylabel('Velocity $u$')
        plt.legend(ncol=4, fontsize=7)
        plt.show()

    if input_array.ndim == 3:
        t, h, w = input_array.shape

        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2) = plt.subplots(2, sharey=True, constrained_layout=True)
        # fig.suptitle('Flow Profiles in X- and Y-Direction', fontsize=10)

        ax1.set_title('Flow Profile in X-Direction', fontsize=10)
        ax1.plot(v_steps, input_array[0, int(h/2), :], label='t = 0')
        ax1.plot(v_steps, input_array[int(
            0.01*t), int(h/2), :], label=f't = {int(0.01*t)}')
        ax1.plot(v_steps, input_array[int(
            0.05*t), int(h/2), :], label=f't = {int(0.05*t)}')
        ax1.plot(v_steps, input_array[int(
            0.10*t), int(h/2), :], label=f't = {int(0.1*t)}')
        ax1.plot(v_steps, input_array[int(
            0.25*t), int(h/2), :], label=f't = {int(0.25*t)}')
        ax1.plot(v_steps, input_array[int(
            0.50*t), int(h/2), :], label=f't = {int(0.5*t)}')
        ax1.plot(v_steps, input_array[int(
            0.75*t), int(h/2), :], label=f't = {int(0.75*t)}')
        ax1.plot(v_steps, input_array[-1, int(h/2), :], label=f't = {t}')

        ax2.set_title('Flow Profile in Y-Direction', fontsize=10)
        ax2.plot(v_steps, input_array[0, :, int(h/2)], label='t = 0')
        ax2.plot(v_steps, input_array[int(
            0.01*t), :, int(h/2)], label=f't = {int(0.01*t)}')
        ax2.plot(v_steps, input_array[int(
            0.05*t), :, int(h/2)], label=f't = {int(0.05*t)}')
        ax2.plot(v_steps, input_array[int(
            0.10*t), :, int(h/2)], label=f't = {int(0.1*t)}')
        ax2.plot(v_steps, input_array[int(
            0.25*t), :, int(h/2)], label=f't = {int(0.25*t)}')
        ax2.plot(v_steps, input_array[int(
            0.50*t), :, int(h/2)], label=f't = {int(0.5*t)}')
        ax2.plot(v_steps, input_array[int(
            0.75*t), :, int(h/2)], label=f't = {int(0.75*t)}')
        ax2.plot(v_steps, input_array[-1, :, int(h/2)], label=f't = {t}')

        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.ylabel('Velocity $u$')
        plt.legend(ncol=4, fontsize=7)
        plt.show()

    if input_array.ndim == 4:
        t, d, h, w = input_array.shape

        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, sharey=True, constrained_layout=True)  # sharex=True
        # plt.ylabel('Velocity $u$')
        # fig.suptitle('Flow Profiles in X-,Y- and Z-Direction ($U_{wall}$ in X-Direction)', fontsize=10)

        ax1.set_title('Flow Profile in X-Direction', fontsize=10)
        ax1.plot(v_steps, input_array[0, int(h/2), int(h/2), :], label='t = 0')
        ax1.plot(v_steps, input_array[int(
            0.01*t), int(h/2), int(h/2), :], label=f't = {int(0.01*t)}')
        ax1.plot(v_steps, input_array[int(
            0.05*t), int(h/2), int(h/2), :], label=f't = {int(0.05*t)}')
        ax1.plot(v_steps, input_array[int(
            0.10*t), int(h/2), int(h/2), :], label=f't = {int(0.1*t)}')
        ax1.plot(v_steps, input_array[int(
            0.25*t), int(h/2), int(h/2), :], label=f't = {int(0.25*t)}')
        ax1.plot(v_steps, input_array[int(
            0.50*t), int(h/2), int(h/2), :], label=f't = {int(0.5*t)}')
        ax1.plot(v_steps, input_array[int(
            0.75*t), int(h/2), int(h/2), :], label=f't = {int(0.75*t)}')
        ax1.plot(v_steps, input_array[-1, int(h/2),
                 int(h/2), :], label=f't = {t}')

        ax2.set_title('Flow Profile in Y-Direction', fontsize=10)
        ax2.set_ylabel('Velocity $u$')
        ax2.plot(v_steps, input_array[0, int(h/2), :, int(h/2)], label='t = 0')
        ax2.plot(v_steps, input_array[int(
            0.01*t), int(h/2), :, int(h/2)], label=f't = {int(0.01*t)}')
        ax2.plot(v_steps, input_array[int(
            0.05*t), int(h/2), :, int(h/2)], label=f't = {int(0.05*t)}')
        ax2.plot(v_steps, input_array[int(
            0.10*t), int(h/2), :, int(h/2)], label=f't = {int(0.1*t)}')
        ax2.plot(v_steps, input_array[int(
            0.25*t), int(h/2), :, int(h/2)], label=f't = {int(0.25*t)}')
        ax2.plot(v_steps, input_array[int(
            0.50*t), int(h/2), :, int(h/2)], label=f't = {int(0.5*t)}')
        ax2.plot(v_steps, input_array[int(
            0.75*t), int(h/2), :, int(h/2)], label=f't = {int(0.75*t)}')
        ax2.plot(v_steps, input_array[-1, int(h/2),
                 :, int(h/2)], label=f't = {t}')

        ax3.set_title('Flow Profile in Z-Direction', fontsize=10)
        ax3.plot(v_steps, input_array[0, :, int(h/2), int(h/2)], label='t = 0')
        ax3.plot(v_steps, input_array[int(
            0.01*t), :, int(h/2), int(h/2)], label=f't = {int(0.01*t)}')
        ax3.plot(v_steps, input_array[int(
            0.05*t), :, int(h/2), int(h/2)], label=f't = {int(0.05*t)}')
        ax3.plot(v_steps, input_array[int(
            0.10*t), :, int(h/2), int(h/2)], label=f't = {int(0.1*t)}')
        ax3.plot(v_steps, input_array[int(
            0.25*t), :, int(h/2), int(h/2)], label=f't = {int(0.25*t)}')
        ax3.plot(v_steps, input_array[int(
            0.50*t), :, int(h/2), int(h/2)], label=f't = {int(0.5*t)}')
        ax3.plot(v_steps, input_array[int(
            0.75*t), :, int(h/2), int(h/2)], label=f't = {int(0.75*t)}')
        ax3.plot(v_steps, input_array[-1, :,
                 int(h/2), int(h/2)], label=f't = {t}')
        # fig.legend(loc='lower center', ncol=4)
        # fig.tight_layout()
        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.legend(ncol=4, fontsize=7)
        plt.show()


def compareFlowProfile(title, file_name, prediction_array, prediction_array2, target_array, analytical=0, wall_height=20, u_wall=10, sigma=0.3):

    if prediction_array.ndim == 1:
        h = prediction_array.shape[0]
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        fig, ax1 = plt.subplots(constrained_layout=True)
        ax1.set_title('Flow Profile in Y-Direction')
        ax1.plot(v_steps, prediction_array, label='prediction')
        ax1.plot(v_steps, target_array,  label='target')

        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.ylabel('Velocity $u$')
        plt.legend(loc="best", ncol=2, fontsize=7)
        # fig.tight_layout()

    if prediction_array.ndim == 2:
        h, w = prediction_array.shape
        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2) = plt.subplots(2, sharey=True, constrained_layout=True)

        ax1.set_title('Flow Profile in X-Direction', fontsize=10)
        ax1.plot(v_steps, prediction_array[int(h/2), :], label='prediction')
        ax1.plot(v_steps, target_array[int(h/2), :], label='target')

        ax2.set_title('Flow Profile in Y-Direction', fontsize=10)
        ax2.plot(v_steps, prediction_array[:, int(h/2)], label='prediction')
        ax2.plot(v_steps, target_array[:, int(h/2)], label='target')

        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.ylabel('Velocity $u$')
        plt.legend(loc="best", ncol=2, fontsize=7)
        # fig.tight_layout()

    if prediction_array.ndim == 3:
        d, h, w = prediction_array.shape
        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, sharey=True, constrained_layout=True)  # sharex=True
        # plt.ylabel('Velocity $u$')

        ax1.set_title('Flow Profile in X-Direction', fontsize=10)
        ax1.plot(v_steps, prediction_array[int(
            h/2), int(h/2), :], label='prediction')
        ax1.plot(v_steps, target_array[int(h/2), int(h/2), :], label='target')

        ax2.set_title('Flow Profile in Y-Direction', fontsize=10)
        ax2.set_ylabel('Velocity $u$')
        ax2.plot(v_steps, prediction_array[int(
            h/2), :, int(h/2)], label='prediction')
        ax2.plot(v_steps, target_array[int(h/2), :, int(h/2)], label='target')

        ax3.set_title('Flow Profile in Z-Direction', fontsize=10)
        ax3.plot(v_steps, prediction_array[:, int(
            h/2), int(h/2)], label='prediction')
        ax3.plot(v_steps, target_array[:, int(h/2), int(h/2)], label='target')

        # fig.legend(loc='lower center', ncol=4)
        # fig.tight_layout()
        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 10))
        plt.xlabel('Spatial Dimension')
        plt.legend(loc="best", ncol=2, fontsize=7)
        # fig.tight_layout()

    if prediction_array.ndim == 4:
        c, d, h, w = prediction_array.shape
        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        switch = True
        if switch == True:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                5, sharey=True, constrained_layout=True)  # sharex=True

            ax1.set_xlabel("X")
            ax1.set_ylabel("$u_x$")
            ax1.plot(v_steps, prediction_array[int(0), int(
                h/2), int(h/2), :], ".r", markersize=0.5, label='MAE prediction')
            ax1.plot(v_steps, prediction_array2[int(0), int(
                h/2), int(h/2), :], ".y", markersize=0.5, label='MSE prediction')
            ax1.plot(v_steps, target_array[0, int(
                h/2), int(h/2), :], label='target')

            ax2.set_xlabel("Y")
            ax2.set_ylabel("$u_x$")
            ax2.plot(v_steps, prediction_array[0, int(
                h/2), :, int(h/2)], ".r", markersize=0.5, label='MAE prediction')
            ax2.plot(v_steps, prediction_array2[0, int(
                h/2), :, int(h/2)], ".y", markersize=0.5, label='MSE prediction')
            ax2.plot(v_steps, target_array[0, int(
                h/2), :, int(h/2)], label='target')

            ax3.set_xlabel("Z")
            ax3.set_ylabel("$u_x$")
            ax3.plot(v_steps, prediction_array[0, :, int(
                h/2), int(h/2)], ".r", markersize=0.5, label='MAE prediction')
            ax3.plot(v_steps, prediction_array2[0, :, int(
                h/2), int(h/2)], ".y", markersize=0.5, label='MSE prediction')
            ax3.plot(v_steps, target_array[0, :, int(
                h/2), int(h/2)], label='target')

            ax4.set_xlabel("X")
            ax4.set_ylabel("$u_y$")
            ax4.plot(v_steps, prediction_array[1, :, int(
                h/2), int(h/2)], ".r", markersize=0.5, label='MAE prediction')
            ax4.plot(v_steps, prediction_array2[1, :, int(
                h/2), int(h/2)], ".y", markersize=0.5, label='MSE prediction')
            ax4.plot(v_steps, target_array[1, :, int(
                h/2), int(h/2)], label='target')

            ax5.set_xlabel("X")
            ax5.set_ylabel("$u_z$")
            ax5.plot(v_steps, prediction_array[2, :, int(
                h/2), int(h/2)], ".r", markersize=0.5, label='MAE prediction')
            ax5.plot(v_steps, prediction_array2[2, :, int(
                h/2), int(h/2)], ".y", markersize=0.5, label='MSE prediction')
            ax5.plot(v_steps, target_array[2, :, int(
                h/2), int(h/2)], label='target')

    if isinstance(analytical, np.ndarray):
        ax1.plot(v_steps, analytical[0, int(
            h/2), int(h/2), :], label='analytical')

        ax2.plot(v_steps, analytical[0, int(
            h/2), :, int(h/2)], label='analytical')

        ax3.plot(v_steps, analytical[0, :, int(
            h/2), int(h/2)], label='analytical')

        ax4.plot(v_steps, analytical[1, :, int(
            h/2), int(h/2)], label='analytical')

        ax5.plot(v_steps, analytical[2, :, int(
            h/2), int(h/2)], label='analytical')

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.yticks(range(int(-u_wall), int(u_wall*(2)+6), 10))
    fig.set_size_inches(3.5, 6)
    plt.show()
    fig.savefig(f'Plots/{file_name}_Flow_Profile.svg')


def compareUxFlowProfile(title, file_name, prediction_array, prediction_array2, target_array, analytical=0, wall_height=20, u_wall=10, sigma=0.3):

    c, d, h, w = prediction_array.shape
    v_step = wall_height / (w-1)
    v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

    fig, ax1 = plt.subplots(
        1, sharey=True, constrained_layout=True)  # sharex=True

    ax1.set_xlabel("Y")
    ax1.set_ylabel("$u_x$")
    ax1.plot(v_steps, prediction_array[0, int(
        h/2), :, int(h/2)], ".r", markersize=0.5, label='MAE prediction')
    ax1.plot(v_steps, prediction_array2[0, int(
        h/2), :, int(h/2)], ".y", markersize=0.5, label='MSE prediction')
    ax1.plot(v_steps, target_array[0, int(
        h/2), :, int(h/2)], label='target')
    ax1.plot(v_steps, analytical[0, int(
        h/2), :, int(h/2)], label='analytical')
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.yticks(range(int(-u_wall), int(u_wall*(2)+6), 10))
    fig.set_size_inches(3.5, (0.3+6/5))
    plt.show()
    fig.savefig(f'Plots/{file_name}_Ux_Flow_Profile.svg')


def plotVelocityField(input_array, wall_height=20):
    if input_array.ndim == 2:
        h, w = input_array.shape
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        X1, Y1 = np.meshgrid(v_steps, v_steps)
        u = input_array
        v = np.zeros(shape=u.shape)

        # set color field for better visualisation
        n = -2
        color = np.sqrt(((v-n)/2)*2 + ((u-n)/2)*2)

        # set plot parameters
        # u - velocity component in x-direction
        # v - velocity component in y-direction
        fig, ax = plt.subplots()
        ax.quiver(X1, Y1, u, v, color, alpha=0.75)

        plt.ylabel('Height $z$')
        plt.xlabel('Depth $x$')
        plt.title('The Startup Couette Velocity Field (Cross-Section)')
        plt.show()

    if input_array.ndim == 3:
        d, h, w = input_array.shape
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        X1, Y1 = np.meshgrid(v_steps, v_steps)
        u = input_array[int(0.5*d)]
        v = np.zeros(shape=u.shape)

        # set color field for better visualisation
        n = -2
        color = np.sqrt(((v-n)/2)*2 + ((u-n)/2)*2)

        # set plot parameters
        # u - velocity component in x-direction
        # v - velocity component in y-direction
        fig, ax = plt.subplots()
        ax.quiver(X1, Y1, u, v, color, alpha=0.75)

        plt.ylabel('Height $z$')
        plt.xlabel('Depth $x$')
        plt.title('The Startup Couette Velocity Field (Cross-Section)')
        plt.show()


def compareVelocityField(prediction_array, target_array, wall_height=20):
    if prediction_array.ndim == 3:
        t, h, w = prediction_array.shape
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        X1, Y1 = np.meshgrid(v_steps, v_steps)

        for i in range(t):
            # set plot parameters
            # u - velocity component in x-direction
            # v - velocity component in y-direction
            u_pred = prediction_array[i]
            u_targ = target_array[i]
            v = np.zeros(shape=u_pred.shape)

            # set color field for better visualisation
            n = -2
            color = np.sqrt(((v-n)/2)*2 + ((u_pred-n)/2)*2)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.quiver(X1, Y1, u_pred, v, color, alpha=0.75)
            ax1.set_aspect('equal')
            ax1.set_title('predicted')
            ax1.set_xlabel('Depth $x$')
            ax1.set_ylabel('Height $z$')

            ax2.quiver(X1, Y1, u_targ, v, color, alpha=0.75)
            ax2.set_aspect('equal')
            ax2.set_title('noisy analytical')
            ax2.set_xlabel('Depth $x$')
            fig.suptitle('The Startup Couette Velocity Field (Cross-Section)')
            plt.show()
            fig.savefig(f'pred_vs_noisy_target_v_field_3e-1_{i}.svg')

    if prediction_array.ndim == 4:
        t, d, h, w = prediction_array.shape
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        X1, Y1 = np.meshgrid(v_steps, v_steps)

        for i in range(t):
            # set plot parameters
            # u - velocity component in x-direction
            # v - velocity component in y-direction
            u_pred = prediction_array[i]
            u_targ = target_array[i]
            v = np.zeros(shape=u_pred.shape)

            # set color field for better visualisation
            n = -2
            color = np.sqrt(((v-n)/2)*2 + ((u_pred-n)/2)*2)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.quiver(X1, Y1, u_pred, v, color, alpha=0.75)
            ax1.set_aspect('equal')
            ax1.set_title('predicted')
            ax1.set_xlabel('Depth $x$')
            ax1.set_ylabel('Height $z$')

            ax2.quiver(X1, Y1, u_targ, v, color, alpha=0.75)
            ax2.set_aspect('equal')
            ax2.set_title('noisy analytical')
            ax2.set_xlabel('Depth $x$')
            fig.suptitle('The Startup Couette Velocity Field (Cross-Section)')
            plt.show()
            fig.savefig(f'pred_vs_noisy_target_v_field_3e-1_{i}.svg')


def colorMap(u_vector):
    len_u = len(u_vector)
    d, h2, w2 = u_vector[0].shape
    v_step = w2 / (h2-1)
    v_steps = np.arange(0, w2 + v_step, v_step).tolist()
    X, Y, Z = np.meshgrid(v_steps, v_steps, v_steps)
    counter = 1
    # Creating color map
    cm = plt.cm.get_cmap('YlGnBu')

    # Creating figure
    fig = plt.figure()
    # fig.suptitle('3D Analytical Couette Flow Sample of $u_x$ at $T=98$',
    #             fontsize=10, fontweight='bold')

    # Creating subplots
    for u in u_vector:
        ax = fig.add_subplot(len_u, 1, counter, projection='3d')
        sc = ax.scatter3D(Z, Y, X, c=u, alpha=0.8, marker='.',
                          s=0.25, vmin=0, vmax=10, cmap=cm)
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
        counter += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    # fig.set_size_inches(3.5, 2)
    plt.show()
    fig.savefig('Plots/Sample_Volume.png')

    '''
    U_1 = u_1
    U_2 = u_2
    U_3 = u_3
    # U_3 = target[2, 0, :, :, :]
    d, h2, w2 = u_1.shape
    v_step = w2 / (h2-1)
    v_steps = np.arange(0, w2 + v_step, v_step).tolist()
    X, Y, Z = np.meshgrid(v_steps, v_steps, v_steps)

    cm = plt.cm.get_cmap('YlGnBu')

    # Creating figure
    fig = plt.figure()
    fig.suptitle('3D Analytical Couette Flow Sample of $u_x$ at $T=98$',
                 fontsize=10, fontweight='bold')

    # Creating plot 1
    ax1 = fig.add_subplot(3, 1, 1, projection='3d')
    sc = ax1.scatter3D(Z, Y, X, c=U_1, alpha=0.8, marker='.',
                       s=0.25, vmin=-4., vmax=14., cmap=cm)
    ax1.set_xlabel("X", fontsize=7, fontweight='bold')
    ax1.set_ylabel("Z", fontsize=7, fontweight='bold')
    ax1.set_zlabel("Y", fontsize=7, fontweight='bold')
    ax1.xaxis.labelpad = -10
    ax1.yaxis.labelpad = -10
    ax1.zaxis.labelpad = -10
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.grid(False)
    # plt.colorbar(sc, orientation="vertical", fraction=0.07, location='left')

    # Creating plot 2
    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    sc = ax2.scatter3D(Z, Y, X, c=U_2, alpha=0.8, marker='.',
                       s=0.25, vmin=-4., vmax=14.,  cmap=cm)
    ax2.set_xlabel("X", fontsize=7, fontweight='bold')
    ax2.set_ylabel("Z", fontsize=7, fontweight='bold')
    ax2.set_zlabel("Y", fontsize=7, fontweight='bold')
    ax2.xaxis.labelpad = -10
    ax2.yaxis.labelpad = -10
    ax2.zaxis.labelpad = -10
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.grid(False)

    # Creating plot 3
    ax3 = fig.add_subplot(3, 1, 3, projection='3d')
    sc = ax3.scatter3D(Z, Y, X, c=U_3, alpha=0.8, marker='.',
                       s=0.25, vmin=-4., vmax=14.,  cmap=cm)
    ax3.set_xlabel("X", fontsize=7, fontweight='bold')
    ax3.set_ylabel("Z", fontsize=7, fontweight='bold')
    ax3.set_zlabel("Y", fontsize=7, fontweight='bold')
    ax3.xaxis.labelpad = -10
    ax3.yaxis.labelpad = -10
    ax3.zaxis.labelpad = -10
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.grid(False)

    # fig.colorbar(sc, orientation="vertical",
    #              fraction=0.07, location='left')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    fig.set_size_inches(3.5, 3.5)
    plt.show()
    '''
    pass


def colorMap2():
    t = 1000
    u = 10
    w = 20
    n = 2
    v = 31
    sigma = 0.0
    seed = 1
    analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                   nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
    noisy = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                              nu=n, vertical_resolution=v, sigma=0.3, my_seed=seed)
    t, c, d, h2, w2 = analytical.shape
    v_step = w / (h2-1)
    v_steps = np.arange(0, w + v_step, v_step).tolist()
    X, Y, Z = np.meshgrid(v_steps, v_steps, v_steps)
    U = [analytical[98, 0, :, :, :], noisy[98, 0, :, :, :]]
    counter = 0
    cm = plt.cm.get_cmap('YlGnBu')

    fig, axes = plt.subplots(nrows=1, ncols=2)
    for ax in axes.flat:
        ax = plt.axes(projection='3d')
        sc = ax.scatter3D(Z, Y, X, c=U[counter],
                          alpha=0.8, marker='.', s=0.25, cmap=cm)
        ax.set_xlabel("X", fontsize=10, fontweight='bold')
        ax.set_ylabel("Z", fontsize=10, fontweight='bold')
        ax.set_zlabel("Y", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 20])
        ax.set_yticks([0, 20])
        ax.set_zticks([0, 20])
        ax.grid(False)
        counter += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)

    plt.show()


def main():

    pass


if __name__ == "__main__":
    colorMap()
