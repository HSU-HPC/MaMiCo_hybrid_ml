import numpy as np
import matplotlib.pyplot as plt
from couette_solver import my1DCouetteSolver, my2DCouetteSolver, my3DCouetteSolver
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


def compareFlowProfile(prediction_array, target_array, wall_height=20, u_wall=10, sigma=0.3):

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
        plt.show()

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
        plt.show()

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
        plt.show()


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


def main():
    t = 100
    v_res = 63

    my_data1 = my2DCouetteSolver(
        desired_timesteps=t, vertical_resolution=v_res, sigma=0)
    my_data2 = my2DCouetteSolver(
        desired_timesteps=t, vertical_resolution=v_res, sigma=0.5)

    plotFlowProfile(my_data1)
    compareFlowProfile(my_data1[35], my_data2[35])
    pass


if __name__ == "__main__":
    main()
